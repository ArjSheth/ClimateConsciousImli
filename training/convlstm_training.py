from pathlib import Path
from random import randint
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint_dir = Path(f"/storage/anwesha.ghosh_ug2023/ConvLSTM/{time.strftime('%Y%m%d_%H%M%S')}_convlstm_checkpoints")
if not checkpoint_dir.exists():
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path = checkpoint_dir / "convlstm_best.pt"
last_checkpoint_path = checkpoint_dir / "convlstm_last.pt"

regularized_path = Path("/storage/anwesha.ghosh_ug2023/data/no2_regularized_200x200.npz")
if not regularized_path.exists():
    raise FileNotFoundError(f"Could not find {regularized_path}. Run the regularization cell first.")

with np.load(regularized_path, allow_pickle=True) as data:
    regularized_no2_tensor = data["grids"].astype(np.float32)
    regularized_no2_mask_tensor = data["masks"].astype(np.float32)
    regularized_dates = pd.to_datetime(data["datetime"])
    compressed_lat_axis = data["lat_axis"].astype(np.float32)
    compressed_lon_axis = data["lon_axis"].astype(np.float32)

print("Grid tensor shape:", regularized_no2_tensor.shape)
print("Mask tensor shape:", regularized_no2_mask_tensor.shape)
print("Date range:", regularized_dates.min(), "->", regularized_dates.max())
print("Latitude axis length:", len(compressed_lat_axis))
print("Longitude axis length:", len(compressed_lon_axis))


def masked_mean_std(grids, masks):
    observed_values = grids[masks > 0]
    if observed_values.size == 0:
        raise ValueError("No observed values found while computing normalization stats.")

    mean = float(observed_values.mean())
    std = float(observed_values.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def normalize_grid(grid, mean, std):
    normalized = (grid - mean) / std
    return np.nan_to_num(normalized, nan=0.0).astype(np.float32)


lookback = 7
num_days = len(regularized_no2_tensor)
if num_days <= lookback:
    raise ValueError(f"Need more than {lookback} daily grids to build temporal samples.")

train_fraction = 0.7
val_fraction = 0.15
train_end = max(lookback + 1, int(num_days * train_fraction))
val_end = max(train_end + 1, int(num_days * (train_fraction + val_fraction)))
val_end = min(val_end, num_days - 1)

train_mean, train_std = masked_mean_std(
    regularized_no2_tensor[:train_end],
    regularized_no2_mask_tensor[:train_end],
)
print(f"Training normalization mean: {train_mean:.4f}")
print(f"Training normalization std:  {train_std:.4f}")


all_target_indices = np.arange(lookback, num_days)
train_target_indices = all_target_indices[all_target_indices < train_end]
val_target_indices = all_target_indices[(all_target_indices >= train_end) & (all_target_indices < val_end)]
test_target_indices = all_target_indices[all_target_indices >= val_end]
batch_size = 16

class NO2TemporalSequenceDataset(Dataset):
    def __init__(self, grids, masks, dates, lookback, mean, std, target_indices):
        self.grids = grids
        self.masks = masks
        self.dates = dates
        self.lookback = lookback
        self.mean = mean
        self.std = std
        self.target_indices = target_indices

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        target_idx = self.target_indices[idx]
        start_idx = target_idx - self.lookback

        input_grids = np.nan_to_num(self.grids[start_idx:target_idx], nan=0.0).astype(np.float32)
        input_masks = np.nan_to_num(self.masks[start_idx:target_idx], nan=0.0).astype(np.float32)
        input_seq = np.stack([input_grids, input_masks], axis=1).astype(np.float32)

        if self.mean is not None and self.std is not None:
            input_seq[:, 0] = (input_seq[:, 0] - self.mean) / self.std

        target = np.nan_to_num(self.grids[target_idx][None, ...], nan=0.0).astype(np.float32)
        target_mask = np.nan_to_num(self.masks[target_idx][None, ...], nan=0.0).astype(np.float32)

        if self.mean is not None and self.std is not None:
            target = (target - self.mean) / self.std

        return (
            torch.from_numpy(input_seq),
            torch.from_numpy(target),
            torch.from_numpy(target_mask),
        )


convlstm_train_dataset = NO2TemporalSequenceDataset(
    regularized_no2_tensor,
    regularized_no2_mask_tensor,
    regularized_dates,
    lookback,
    train_mean,
    train_std,
    train_target_indices,
)
convlstm_val_dataset = NO2TemporalSequenceDataset(
    regularized_no2_tensor,
    regularized_no2_mask_tensor,
    regularized_dates,
    lookback,
    train_mean,
    train_std,
    val_target_indices,
)
convlstm_test_dataset = NO2TemporalSequenceDataset(
    regularized_no2_tensor,
    regularized_no2_mask_tensor,
    regularized_dates,
    lookback,
    train_mean,
    train_std,
    test_target_indices,
)

convlstm_train_loader = DataLoader(convlstm_train_dataset, batch_size=batch_size, shuffle=True)
convlstm_val_loader = DataLoader(convlstm_val_dataset, batch_size=batch_size, shuffle=False)
convlstm_test_loader = DataLoader(convlstm_test_dataset, batch_size=batch_size, shuffle=False)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        input_gate, forget_gate, output_gate, candidate_gate = torch.chunk(gates, 4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate_gate = torch.tanh(candidate_gate)
        c_next = forget_gate * c_prev + input_gate * candidate_gate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next


class MaskAwareConvLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("Expected input of shape [batch, time, channels, height, width].")

        batch_size, time_steps, _, height, width = x.shape
        h_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=x.device, dtype=x.dtype)
        c_state = torch.zeros_like(h_state)

        for time_index in range(time_steps):
            h_state, c_state = self.cell(x[:, time_index], (h_state, c_state))

        return self.decoder(h_state)


convlstm_model = MaskAwareConvLSTM(input_dim=2, hidden_dim=32).to(device)
print(convlstm_model)

convlstm_sample_seq, convlstm_sample_target, convlstm_sample_mask = convlstm_train_dataset[0]
print("ConvLSTM sample shapes:", convlstm_sample_seq.shape, convlstm_sample_target.shape, convlstm_sample_mask.shape)


def masked_mse(prediction, target, target_mask, eps=1e-6):
    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)
    squared_error = (prediction - target) ** 2 * target_mask
    return squared_error.sum() / target_mask.sum().clamp_min(eps)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_masked_mae = 0.0
    total_masked_points = 0.0

    for x, y, y_mask in loader:
        x = x.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = masked_mse(prediction, y, y_mask)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            masked_abs_error = (prediction - y).abs() * y_mask
            batch_masked_points = y_mask.sum().item()
            total_loss += loss.item() * max(batch_masked_points, 1.0)
            total_masked_mae += masked_abs_error.sum().item()
            total_masked_points += batch_masked_points

    mean_loss = total_loss / max(total_masked_points, 1.0)
    mean_mae = total_masked_mae / max(total_masked_points, 1.0)
    return mean_loss, mean_mae


def evaluate_on_loader(model, loader, mean, std):
    model.eval()
    preds = []
    targets = []
    masks = []

    with torch.no_grad():
        for x, y, y_mask in loader:
            x = x.to(device)
            y = y.to(device)
            y_mask = y_mask.to(device)
            prediction = model(x)

            preds.append(prediction.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(y_mask.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    masks = np.concatenate(masks, axis=0)

    preds = preds * std + mean
    targets = targets * std + mean

    valid = masks > 0
    if not np.any(valid):
        raise ValueError("No valid target cells were found during evaluation.")

    mae = np.mean(np.abs(preds[valid] - targets[valid]))
    rmse = np.sqrt(np.mean((preds[valid] - targets[valid]) ** 2))
    return preds, targets, masks, mae, rmse


torch.manual_seed(42)
np.random.seed(42)

convlstm_optimizer = torch.optim.AdamW(convlstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
convlstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    convlstm_optimizer,
    mode="min",
    patience=3,
    factor=0.5,
)

convlstm_best_val_loss = float("inf")
convlstm_best_state = None
convlstm_history = []
convlstm_num_epochs = 3

if last_checkpoint_path.exists():
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    convlstm_model.load_state_dict(checkpoint["model_state_dict"])
    convlstm_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    convlstm_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    convlstm_best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    convlstm_history = checkpoint.get("history", [])
    print(f"Resumed training from {last_checkpoint_path}")

for epoch in range(1, convlstm_num_epochs + 1):
    convlstm_train_loss, convlstm_train_mae = run_epoch(
        convlstm_model,
        convlstm_train_loader,
        optimizer=convlstm_optimizer,
    )
    convlstm_val_loss, convlstm_val_mae = run_epoch(convlstm_model, convlstm_val_loader)
    convlstm_scheduler.step(convlstm_val_loss)

    convlstm_history.append(
        {
            "epoch": epoch,
            "train_loss": convlstm_train_loss,
            "train_mae": convlstm_train_mae,
            "val_loss": convlstm_val_loss,
            "val_mae": convlstm_val_mae,
        }
    )

    if convlstm_val_loss < convlstm_best_val_loss:
        convlstm_best_val_loss = convlstm_val_loss
        convlstm_best_state = {key: value.detach().cpu().clone() for key, value in convlstm_model.state_dict().items()}

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": convlstm_best_state,
                "optimizer_state_dict": convlstm_optimizer.state_dict(),
                "scheduler_state_dict": convlstm_scheduler.state_dict(),
                "best_val_loss": convlstm_best_val_loss,
                "history": convlstm_history,
                "train_mean": train_mean,
                "train_std": train_std,
                "lookback": lookback,
                "batch_size": batch_size,
            },
            best_checkpoint_path,
        )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": convlstm_model.state_dict(),
            "optimizer_state_dict": convlstm_optimizer.state_dict(),
            "scheduler_state_dict": convlstm_scheduler.state_dict(),
            "best_val_loss": convlstm_best_val_loss,
            "history": convlstm_history,
            "train_mean": train_mean,
            "train_std": train_std,
            "lookback": lookback,
            "batch_size": batch_size,
        },
        last_checkpoint_path,
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train loss {convlstm_train_loss:.4f} | train mae {convlstm_train_mae:.4f} | "
        f"val loss {convlstm_val_loss:.4f} | val mae {convlstm_val_mae:.4f}"
    )


if convlstm_best_state is not None:
    convlstm_model.load_state_dict(convlstm_best_state)
elif best_checkpoint_path.exists():
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    convlstm_model.load_state_dict(checkpoint["model_state_dict"])
    convlstm_best_val_loss = checkpoint.get("best_val_loss", convlstm_best_val_loss)

torch.save(
    {
        "model_state_dict": convlstm_model.state_dict(),
        "best_val_loss": convlstm_best_val_loss,
        "history": convlstm_history,
        "train_mean": train_mean,
        "train_std": train_std,
        "lookback": lookback,
        "batch_size": batch_size,
    },
    best_checkpoint_path,
)

convlstm_history_df = pd.DataFrame(convlstm_history)
convlstm_history_df

convlstm_preds, convlstm_targets, convlstm_masks, convlstm_test_mae, convlstm_test_rmse = evaluate_on_loader(
    convlstm_model,
    convlstm_test_loader,
    train_mean,
    train_std,
)
print(f"ConvLSTM Test MAE:  {convlstm_test_mae:.4f}")
print(f"ConvLSTM Test RMSE: {convlstm_test_rmse:.4f}")

# Show 8 random samples (target, prediction, mask) instead of a single one
num_samples = 8
indices = torch.randperm(len(convlstm_test_dataset))[:num_samples].tolist()

fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9), constrained_layout=True)
for col, idx in enumerate(indices):
    sample_seq, sample_target, sample_mask = convlstm_test_dataset[idx]
    with torch.no_grad():
        sample_pred = convlstm_model(sample_seq.unsqueeze(0).to(device)).cpu().squeeze(0).squeeze(0)

    sample_target = sample_target.squeeze(0) * train_std + train_mean
    sample_prediction = sample_pred * train_std + train_mean
    sample_mask_2d = sample_mask.squeeze(0)

    ax_t = axes[0, col]
    im_t = ax_t.imshow(sample_target.numpy(), cmap="viridis")
    ax_t.set_title(f"Target #{idx}")
    ax_t.axis("off")

    ax_p = axes[1, col]
    im_p = ax_p.imshow(sample_prediction.numpy(), cmap="viridis")
    ax_p.set_title("Prediction")
    ax_p.axis("off")

    ax_m = axes[2, col]
    im_m = ax_m.imshow(sample_mask_2d.numpy(), cmap="magma", vmin=0, vmax=1)
    ax_m.set_title("Mask")
    ax_m.axis("off")

    fig.colorbar(im_t, ax=ax_t, shrink=0.6)
    fig.colorbar(im_p, ax=ax_p, shrink=0.6)
    fig.colorbar(im_m, ax=ax_m, shrink=0.6)

# Save the multi-sample plot
plot_path = checkpoint_dir / "sample_predictions_8.png"
plt.savefig(plot_path)
plt.close()



if not convlstm_history_df.empty:
    plt.figure(figsize=(8, 4))
    plt.plot(convlstm_history_df["epoch"], convlstm_history_df["train_loss"], label="train")
    plt.plot(convlstm_history_df["epoch"], convlstm_history_df["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Masked MSE")
    plt.title("ConvLSTM training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    # Save the second plot (training curve) instead of showing it
    training_curve_path = checkpoint_dir / "training_curve_epoch_loss.png"
    plt.savefig(training_curve_path)
    plt.close()

