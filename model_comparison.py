import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# Update these paths to your saved checkpoints. Use None to skip a model.
checkpoint_paths = {
    "cnn": "/home/vishnu/Desktop/Ashoka/sem_4/IML/anu/weeeeeeeeeeeeeee/cnn/cnn_last.pt",
    "convlstm": "/home/vishnu/Desktop/Ashoka/sem_4/IML/anu/weeeeeeeeeeeeeee/clstm/convlstm_last.pt",
    "unet": "/home/vishnu/Desktop/Ashoka/sem_4/IML/anu/weeeeeeeeeeeeeee/unet/unet_last.pt",
    "weather": "/home/vishnu/Desktop/Ashoka/sem_4/IML/anu/weeeeeeeeeeeeeee/weather/weather_aware_last.pt",
}


# Ensure required globals exist before running.
required_globals = [
    "test_loader",
    "test_dataset",
    "convlstm_test_loader",
    "convlstm_test_dataset",
    "train_mean",
    "train_std",
]
missing_globals = [name for name in required_globals if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Missing required variables: "
        + ", ".join(missing_globals)
        + ". Run the data prep/training cells first."
    )


if "device" not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if "MaskAwareCNN" not in globals():
    raise RuntimeError("MaskAwareCNN not found. Run the CNN definition cell first.")
if "MaskAwareConvLSTM" not in globals():
    raise RuntimeError("MaskAwareConvLSTM not found. Run the ConvLSTM definition cell first.")
if "MaskAwareUNet" not in globals():
    raise RuntimeError("MaskAwareUNet not found. Run the UNet definition cell first.")
if "WeatherAwareNO2FusionNet" not in globals():
    raise RuntimeError("WeatherAwareNO2FusionNet not found. Run the weather-aware model cell first.")


# Helper utilities

def _normalize_checkpoint_path(path_value):
    if path_value is None:
        return None
    path_value = str(path_value).strip()
    return path_value if path_value else None


def _load_checkpoint_state(checkpoint_path):
    if checkpoint_path is None:
        return None
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _move_to_device(obj, target_device):
    if isinstance(obj, torch.Tensor):
        return obj.to(target_device)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, target_device) for key, value in obj.items()}
    return obj


def _to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj


def _ensure_no2_batch(x):
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def _ensure_seq_batch(x):
    if x.ndim == 4:
        return x.unsqueeze(0)
    return x


def _ensure_weather_inputs(inputs):
    no2_day = _ensure_no2_batch(inputs["no2_day"])
    no2_mask = _ensure_no2_batch(inputs["no2_mask"])
    temp_seq = _ensure_seq_batch(inputs["temp_seq"])
    wind_seq = _ensure_seq_batch(inputs["wind_seq"])
    return {
        "no2_day": no2_day,
        "no2_mask": no2_mask,
        "temp_seq": temp_seq,
        "wind_seq": wind_seq,
    }


def compute_metrics(pred, target, mask):
    if pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "PSNR": np.nan}
    error = pred - target
    abs_error = np.abs(error)[mask]
    sq_error = (error ** 2)[mask]

    if abs_error.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "PSNR": np.nan}

    mae = float(abs_error.mean())
    rmse = float(np.sqrt(sq_error.mean()))

    target_valid = target[mask]
    data_range = float(target_valid.max() - target_valid.min()) if target_valid.size else 0.0
    if data_range <= 0 or rmse == 0:
        psnr = np.nan
    else:
        psnr = 20.0 * math.log10(data_range / rmse)

    return {"MAE": mae, "RMSE": rmse, "PSNR": psnr}


def evaluate_model(model, loader, mean=None, std=None, max_scatter=6000):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_count = 0
    target_sum = 0.0
    target_sq_sum = 0.0
    target_abs_sum = 0.0
    target_min = np.inf
    target_max = -np.inf
    scatter_true = []
    scatter_pred = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x, y, y_mask = batch
                x = _move_to_device(x, device)
                y = _move_to_device(y, device)
                y_mask = _move_to_device(y_mask, device)
                prediction = model(x)
            elif isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], dict):
                inputs, targets = batch
                inputs = _move_to_device(inputs, device)
                inputs = _ensure_weather_inputs(inputs)
                targets = _move_to_device(targets, device)
                prediction = model(inputs)
                y = targets.get("next_no2")
                if y is None:
                    raise KeyError("Weather-aware targets must include 'next_no2'.")
                y_mask = torch.isfinite(y).float()
            else:
                raise ValueError("Unsupported batch format for evaluation.")

            pred_np = _to_numpy(prediction)
            target_np = _to_numpy(y)
            mask_np = _to_numpy(y_mask)

            if mean is not None and std is not None:
                pred_np = pred_np * float(std) + float(mean)
                target_np = target_np * float(std) + float(mean)

            valid = mask_np > 0
            if not np.any(valid):
                continue

            diff = pred_np - target_np
            total_abs += np.abs(diff)[valid].sum()
            total_sq += (diff ** 2)[valid].sum()
            total_count += int(valid.sum())

            valid_targets = target_np[valid]
            target_sum += float(valid_targets.sum())
            target_sq_sum += float((valid_targets ** 2).sum())
            target_abs_sum += float(np.abs(valid_targets).sum())

            target_min = min(target_min, float(np.nanmin(valid_targets)))
            target_max = max(target_max, float(np.nanmax(valid_targets)))

            if max_scatter and len(scatter_true) < max_scatter:
                flat_valid = np.flatnonzero(valid.ravel())
                if flat_valid.size > 0:
                    take = min(max_scatter - len(scatter_true), flat_valid.size)
                    choice = np.random.choice(flat_valid, size=take, replace=False)
                    flat_pred = pred_np.ravel()[choice]
                    flat_true = target_np.ravel()[choice]
                    scatter_pred.append(flat_pred)
                    scatter_true.append(flat_true)

    if total_count == 0:
        return None, None

    mae = total_abs / total_count
    rmse = math.sqrt(total_sq / total_count)
    data_range = target_max - target_min
    psnr = np.nan if data_range <= 0 or rmse == 0 else 20.0 * math.log10(data_range / rmse)

    target_mean = target_sum / total_count
    target_var = max(target_sq_sum / total_count - target_mean ** 2, 0.0)
    target_std = math.sqrt(target_var)
    target_mean_abs = target_abs_sum / total_count

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR": psnr,
        "MeanAbsTarget": target_mean_abs,
        "TargetMean": target_mean,
        "TargetStd": target_std,
        "TargetRange": data_range,
        "MAE_pct_mean_abs": 100.0 * mae / target_mean_abs if target_mean_abs > 0 else np.nan,
        "RMSE_pct_std": 100.0 * rmse / target_std if target_std > 0 else np.nan,
        "RMSE_pct_range": 100.0 * rmse / data_range if data_range > 0 else np.nan,
    }

    scatter_true = np.concatenate(scatter_true) if scatter_true else np.array([])
    scatter_pred = np.concatenate(scatter_pred) if scatter_pred else np.array([])

    return metrics, (scatter_true, scatter_pred)


def _build_weather_splits(frame, train_frac=0.7, val_frac=0.15):
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"]).dt.normalize()
    work = work.sort_values("datetime").reset_index(drop=True)
    unique_dates = work["datetime"].drop_duplicates().to_list()
    if len(unique_dates) < 3:
        raise ValueError("Not enough dates to build train/val/test splits for weather data.")

    train_cut = max(1, int(len(unique_dates) * train_frac))
    val_cut = max(train_cut + 1, int(len(unique_dates) * (train_frac + val_frac)))
    val_cut = min(val_cut, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])

    train_df = work[work["datetime"].isin(train_dates)].reset_index(drop=True)
    val_df = work[work["datetime"].isin(val_dates)].reset_index(drop=True)
    test_df = work[work["datetime"].isin(test_dates)].reset_index(drop=True)
    return train_df, val_df, test_df


# Build models and load checkpoints
sample_x, sample_y, sample_mask = test_dataset[0]
cnn_in_channels = int(sample_x.shape[0])

sample_seq, _, _ = convlstm_test_dataset[0]
convlstm_input_dim = int(sample_seq.shape[1])

cnn_model = MaskAwareCNN(in_channels=cnn_in_channels).to(device)
convlstm_model = MaskAwareConvLSTM(input_dim=convlstm_input_dim).to(device)
unet_model = MaskAwareUNet(in_channels=cnn_in_channels).to(device)
weather_model = WeatherAwareNO2FusionNet().to(device)

for key, model in (
    ("cnn", cnn_model),
    ("convlstm", convlstm_model),
    ("unet", unet_model),
    ("weather", weather_model),
):
    ckpt = _normalize_checkpoint_path(checkpoint_paths.get(key))
    if ckpt is None:
        continue
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
    try:
        state = _load_checkpoint_state(ckpt)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        checkpoint = torch.load(Path(ckpt).expanduser(), map_location=device, weights_only=False)
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)


# Prepare weather-aware test loader (if available)
weather_test_loader = None
weather_test_dataset = None
if "weather_preview_full_df" in globals():
    _, _, weather_test_df = _build_weather_splits(weather_preview_full_df)
    weather_test_dataset = WeatherAwareNO2Dataset(weather_test_df)
    weather_test_loader = torch.utils.data.DataLoader(weather_test_dataset, batch_size=4, shuffle=False)


# Evaluate metrics
results = []
scatter_data = {}

metrics, scatter = evaluate_model(cnn_model, test_loader, mean=train_mean, std=train_std)
if metrics:
    results.append({"Model": "CNN", **metrics})
    scatter_data["CNN"] = scatter

metrics, scatter = evaluate_model(unet_model, test_loader, mean=train_mean, std=train_std)
if metrics:
    results.append({"Model": "UNet", **metrics})
    scatter_data["UNet"] = scatter

metrics, scatter = evaluate_model(convlstm_model, convlstm_test_loader, mean=train_mean, std=train_std)
if metrics:
    results.append({"Model": "ConvLSTM", **metrics})
    scatter_data["ConvLSTM"] = scatter

if weather_test_loader is not None:
    metrics, scatter = evaluate_model(weather_model, weather_test_loader)
    if metrics:
        results.append({"Model": "Weather-aware", **metrics})
        scatter_data["Weather-aware"] = scatter

metrics_df = pd.DataFrame(results).sort_values("RMSE")
summary_cols = [
    "Model",
    "MAE",
    "RMSE",
    "PSNR",
    "MeanAbsTarget",
    "TargetStd",
    "TargetRange",
    "MAE_pct_mean_abs",
    "RMSE_pct_std",
    "RMSE_pct_range",
]

print("Overall comparison table:")
print(metrics_df[summary_cols].round(4).to_string(index=False))

if not metrics_df.empty and "Weather-aware" in metrics_df["Model"].values:
    weather_row = metrics_df.loc[metrics_df["Model"] == "Weather-aware"].iloc[0]
    print("\nWeather-aware model summary:")
    print(f"MAE: {weather_row['MAE']:.6f}")
    print(f"RMSE: {weather_row['RMSE']:.6f}")
    print(f"PSNR: {weather_row['PSNR']:.4f}")
    print(f"Typical target magnitude (mean abs): {weather_row['MeanAbsTarget']:.6f}")
    print(f"Target std: {weather_row['TargetStd']:.6f}")
    print(f"Target range: {weather_row['TargetRange']:.6f}")
    print(f"MAE as % of mean abs target: {weather_row['MAE_pct_mean_abs']:.2f}%")
    print(f"RMSE as % of target std: {weather_row['RMSE_pct_std']:.2f}%")
    print(f"RMSE as % of target range: {weather_row['RMSE_pct_range']:.2f}%")

if not metrics_df.empty:
    signal_scale = metrics_df[["Model", "MeanAbsTarget", "TargetStd", "TargetRange"]].round(6)
    print("\nTarget scale by model:")
    print(signal_scale.to_string(index=False))


# Scatter plots: actual vs predicted
if scatter_data:
    model_names = list(scatter_data.keys())
    fig, axes = plt.subplots(1, len(model_names), figsize=(5 * len(model_names), 4), constrained_layout=True)
    if len(model_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, model_names):
        y_true, y_pred = scatter_data[name]
        if y_true.size == 0:
            ax.set_title(f"{name} (no samples)")
            ax.axis("off")
            continue
        ax.scatter(y_true, y_pred, s=5, alpha=0.3)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)
    plt.show()


# Grid comparisons for CNN/UNet/ConvLSTM on shared test samples.
# Layout: 3 lookback days + target + 3 model predictions.
rng = np.random.default_rng()
num_samples = 3
max_index = min(len(test_dataset), len(convlstm_test_dataset))
if max_index > 0:
    sample_indices = rng.choice(np.arange(max_index), size=min(num_samples, max_index), replace=False)
    col_titles = ["t-3", "t-2", "t-1", "Target", "CNN", "UNet", "ConvLSTM"]

    fig, axes = plt.subplots(
        len(sample_indices),
        len(col_titles),
        figsize=(4 * len(col_titles), 3.5 * len(sample_indices)),
        constrained_layout=True,
    )
    if len(sample_indices) == 1:
        axes = np.array([axes])

    for row_idx, idx in enumerate(sample_indices):
        x_cnn, y_cnn, mask_cnn = test_dataset[idx]
        x_seq, y_seq, mask_seq = convlstm_test_dataset[idx]

        with torch.no_grad():
            pred_cnn = cnn_model(x_cnn.unsqueeze(0).to(device)).cpu().squeeze(0)
            pred_unet = unet_model(x_cnn.unsqueeze(0).to(device)).cpu().squeeze(0)
            pred_convlstm = convlstm_model(x_seq.unsqueeze(0).to(device)).cpu().squeeze(0)

        lookback_grids = x_cnn[:lookback].cpu().numpy()
        lookback_grids = lookback_grids[-3:] * train_std + train_mean
        target = (y_cnn.squeeze(0) * train_std + train_mean).numpy()
        pred_cnn = (pred_cnn.squeeze(0).numpy() * train_std + train_mean)
        pred_unet = (pred_unet.squeeze(0).numpy() * train_std + train_mean)
        pred_convlstm = (pred_convlstm.squeeze(0).numpy() * train_std + train_mean)

        vmin = np.nanmin([target, pred_cnn, pred_unet, pred_convlstm])
        vmax = np.nanmax([target, pred_cnn, pred_unet, pred_convlstm])

        grids = [lookback_grids[0], lookback_grids[1], lookback_grids[2], target, pred_cnn, pred_unet, pred_convlstm]
        cmaps = ["viridis", "viridis", "viridis", "viridis", "magma", "magma", "magma"]

        for col_idx, (grid, cmap) in enumerate(zip(grids, cmaps)):
            ax = axes[row_idx, col_idx]
            ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
            ax.axis("off")

    plt.show()


# Weather-aware sample comparison (separate split)
if weather_test_dataset is not None and len(weather_test_dataset) > 0:
    idx = int(rng.integers(0, len(weather_test_dataset)))
    inputs, targets = weather_test_dataset[idx]
    inputs = _ensure_weather_inputs(_move_to_device(inputs, device))
    with torch.no_grad():
        pred = weather_model(inputs).cpu().squeeze(0)

    target = targets["next_no2"]
    if target.ndim == 4:
        target = target.squeeze(0)
    if target.ndim == 3:
        target = target.squeeze(0)
    target = target.numpy()

    pred = pred.squeeze(0).numpy()
    mask = np.isfinite(target).astype(np.float32)

    vmin = np.nanmin([target, pred])
    vmax = np.nanmax([target, pred])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Weather-aware target")
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title("Weather-aware prediction")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Target mask")
    axes[2].axis("off")
    plt.show()