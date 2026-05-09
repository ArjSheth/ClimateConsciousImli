from pathlib import Path
import time

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

checkpoint_dir = Path(f"/storage/anwesha.ghosh_ug2023/CNN/{time.strftime('%Y%m%d_%H%M%S')}_cnn_checkpoints")
if not checkpoint_dir.exists():
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path = checkpoint_dir / "cnn_best.pt"
last_checkpoint_path = checkpoint_dir / "cnn_last.pt"


regularized_path = Path("/storage/anwesha.ghosh_ug2023/data/no2_regularized_200x200.npz")
print(f"Loading regularized data from: {regularized_path}")

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

if len(train_target_indices) == 0 or len(val_target_indices) == 0 or len(test_target_indices) == 0:
	raise ValueError(
		"One of the train/validation/test splits is empty. "
		"Check the regularized dataset size and the lookback window."
	)

batch_size = 16


class NO2TemporalGridDataset(Dataset):
	def __init__(self, grids, masks, dates, lookback, mean, std, target_indices):
		self.grids = grids
		self.masks = masks
		self.dates = pd.to_datetime(dates)
		self.lookback = lookback
		self.mean = mean
		self.std = std
		self.target_indices = np.asarray(target_indices, dtype=np.int64)

	def __len__(self):
		return len(self.target_indices)

	def __getitem__(self, item):
		target_idx = int(self.target_indices[item])
		start_idx = target_idx - self.lookback

		input_grids = self.grids[start_idx:target_idx]
		input_masks = self.masks[start_idx:target_idx]
		target_grid = self.grids[target_idx]
		target_mask = self.masks[target_idx]

		input_grids = normalize_grid(input_grids, self.mean, self.std)
		target_grid = normalize_grid(target_grid, self.mean, self.std)

		x = np.concatenate([input_grids, input_masks], axis=0).astype(np.float32)
		y = target_grid[np.newaxis, ...].astype(np.float32)
		y_mask = target_mask[np.newaxis, ...].astype(np.float32)

		return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(y_mask)


cnn_train_dataset = NO2TemporalGridDataset(
	regularized_no2_tensor,
	regularized_no2_mask_tensor,
	regularized_dates,
	lookback,
	train_mean,
	train_std,
	train_target_indices,
)
cnn_val_dataset = NO2TemporalGridDataset(
	regularized_no2_tensor,
	regularized_no2_mask_tensor,
	regularized_dates,
	lookback,
	train_mean,
	train_std,
	val_target_indices,
)
cnn_test_dataset = NO2TemporalGridDataset(
	regularized_no2_tensor,
	regularized_no2_mask_tensor,
	regularized_dates,
	lookback,
	train_mean,
	train_std,
	test_target_indices,
)

cnn_train_loader = DataLoader(cnn_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
cnn_val_loader = DataLoader(cnn_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
cnn_test_loader = DataLoader(cnn_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class MaskAwareCNN(nn.Module):
	def __init__(self, in_channels, hidden_channels=64):
		super().__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(hidden_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(hidden_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
			nn.BatchNorm2d(hidden_channels // 2),
			nn.ReLU(inplace=True),
		)
		self.head = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)

	def forward(self, x):
		features = self.encoder(x)
		return self.head(features)


def masked_mse(prediction, target, target_mask, eps=1e-6):
	prediction = torch.nan_to_num(prediction)
	target = torch.nan_to_num(target)
	squared_error = (prediction - target) ** 2 * target_mask
	return squared_error.sum() / target_mask.sum().clamp_min(eps)


model = MaskAwareCNN(in_channels=lookback * 2).to(device)
print(model)


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

	if not preds:
		raise ValueError("No batches were produced during evaluation.")

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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

best_val_loss = float("inf")
best_state = None
history = []
epochs_to_run = 15

if last_checkpoint_path.exists():
	checkpoint = torch.load(last_checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
	best_val_loss = checkpoint.get("best_val_loss", float("inf"))
	history = checkpoint.get("history", [])
	print(f"Resumed training from {last_checkpoint_path}")

start_epoch = history[-1]["epoch"] + 1 if history else 1
end_epoch = start_epoch + epochs_to_run - 1

for epoch in range(start_epoch, end_epoch + 1):
	train_loss, train_mae = run_epoch(model, cnn_train_loader, optimizer=optimizer)
	val_loss, val_mae = run_epoch(model, cnn_val_loader)
	scheduler.step(val_loss)

	history.append(
		{
			"epoch": epoch,
			"train_loss": train_loss,
			"train_mae": train_mae,
			"val_loss": val_loss,
			"val_mae": val_mae,
		}
	)

	if val_loss < best_val_loss:
		best_val_loss = val_loss
		best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

		torch.save(
			{
				"epoch": epoch,
				"model_state_dict": best_state,
				"optimizer_state_dict": optimizer.state_dict(),
				"scheduler_state_dict": scheduler.state_dict(),
				"best_val_loss": best_val_loss,
				"history": history,
				"train_mean": train_mean,
				"train_std": train_std,
				"lookback": lookback,
				"batch_size": batch_size,
				"regularized_path": str(regularized_path),
			},
			best_checkpoint_path,
		)

	torch.save(
		{
			"epoch": epoch,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"scheduler_state_dict": scheduler.state_dict(),
			"best_val_loss": best_val_loss,
			"history": history,
			"train_mean": train_mean,
			"train_std": train_std,
			"lookback": lookback,
			"batch_size": batch_size,
			"regularized_path": str(regularized_path),
		},
		last_checkpoint_path,
	)

	print(
		f"Epoch {epoch:02d} | "
		f"train loss {train_loss:.4f} | train mae {train_mae:.4f} | "
		f"val loss {val_loss:.4f} | val mae {val_mae:.4f}"
	)


if best_state is not None:
	model.load_state_dict(best_state)
elif best_checkpoint_path.exists():
	checkpoint = torch.load(best_checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	best_val_loss = checkpoint.get("best_val_loss", best_val_loss)

torch.save(
	{
		"model_state_dict": model.state_dict(),
		"best_val_loss": best_val_loss,
		"history": history,
		"train_mean": train_mean,
		"train_std": train_std,
		"lookback": lookback,
		"batch_size": batch_size,
		"regularized_path": str(regularized_path),
	},
	best_checkpoint_path,
)

history_df = pd.DataFrame(history)
print(history_df)

preds, targets, masks, test_mae, test_rmse = evaluate_on_loader(
	model,
	cnn_test_loader,
	train_mean,
	train_std,
)
print(f"CNN Test MAE:  {test_mae:.4f}")
print(f"CNN Test RMSE: {test_rmse:.4f}")

num_samples = 8
indices = torch.randperm(len(cnn_test_dataset))[:num_samples].tolist()

fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9), constrained_layout=True)
for col, idx in enumerate(indices):
	sample_x, sample_y, sample_mask = cnn_test_dataset[idx]
	with torch.no_grad():
		sample_pred = model(sample_x.unsqueeze(0).to(device)).cpu().squeeze(0).squeeze(0)

	sample_target = sample_y.squeeze(0) * train_std + train_mean
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

plot_path = checkpoint_dir / "sample_predictions_8.png"
plt.savefig(plot_path)
plt.close()

if not history_df.empty:
	plt.figure(figsize=(8, 4))
	plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
	plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
	plt.xlabel("Epoch")
	plt.ylabel("Masked MSE")
	plt.title("CNN training curve")
	plt.legend()
	plt.grid(alpha=0.3)
	training_curve_path = checkpoint_dir / "training_curve_epoch_loss.png"
	plt.savefig(training_curve_path)
	plt.close()
