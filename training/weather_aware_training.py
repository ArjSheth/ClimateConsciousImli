from pathlib import Path
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset_path = 	Path("/storage/anwesha.ghosh_ug2023/data/weather_aware_dataset.npz"),

print(f"Loading weather-aware dataset from {dataset_path}")

with np.load(dataset_path, allow_pickle=True) as data:
	dates = pd.to_datetime(data["datetime"]).to_numpy(dtype="datetime64[ns]")
	no2_day = data["no2_day"].astype(np.float32)
	no2_mask = data["no2_mask"].astype(np.float32)
	temp_seq = data["temp_seq"].astype(np.float32)
	wind_seq = data["wind_seq"].astype(np.float32)
	next_no2 = data["next_no2"].astype(np.float32)

if dates.ndim != 1:
	dates = dates.reshape(-1)

order = np.argsort(dates)
dates = dates[order]
no2_day = no2_day[order]
no2_mask = no2_mask[order]
temp_seq = temp_seq[order]
wind_seq = wind_seq[order]
next_no2 = next_no2[order]

print("Dataset shapes:")
print("dates:", dates.shape)
print("no2_day:", no2_day.shape)
print("no2_mask:", no2_mask.shape)
print("temp_seq:", temp_seq.shape)
print("wind_seq:", wind_seq.shape)
print("next_no2:", next_no2.shape)


def masked_mean_std(grids, masks):
	observed_values = grids[masks > 0]
	observed_values = observed_values[~np.isnan(observed_values)]
	if observed_values.size == 0:
		raise ValueError("No observed values found while computing NO2 normalization stats.")

	mean = float(observed_values.mean())
	std = float(observed_values.std())
	if std < 1e-6:
		std = 1.0
	return mean, std


num_samples = len(dates)
if num_samples == 0:
	raise ValueError("Loaded dataset has zero samples.")

train_fraction = 0.7
val_fraction = 0.15
train_end = max(1, int(num_samples * train_fraction))
val_end = max(train_end + 1, int(num_samples * (train_fraction + val_fraction)))
val_end = min(val_end, num_samples - 1)

train_no2 = no2_day[:train_end]
train_mask = no2_mask[:train_end]
no2_mean, no2_std = masked_mean_std(train_no2, train_mask)
print(f"NO2 normalization mean: {no2_mean:.4f}")
print(f"NO2 normalization std:  {no2_std:.4f}")

train_temp = temp_seq[:train_end]
temp_mean = float(np.nanmean(train_temp))
temp_std = float(np.nanstd(train_temp))
if temp_std < 1e-6:
	temp_std = 1.0
print(f"TEMP normalization mean: {temp_mean:.4f}")
print(f"TEMP normalization std:  {temp_std:.4f}")

train_wind = wind_seq[:train_end]
wind_mean = np.nanmean(train_wind, axis=(0, 1, 3, 4))
wind_std = np.nanstd(train_wind, axis=(0, 1, 3, 4))
wind_std = np.where(wind_std < 1e-6, 1.0, wind_std)
print("WIND normalization mean:", wind_mean)
print("WIND normalization std:", wind_std)


class WeatherAwareNO2Dataset(Dataset):
	def __init__(
		self,
		dates,
		no2_day,
		no2_mask,
		temp_seq,
		wind_seq,
		next_no2,
		no2_mean,
		no2_std,
		temp_mean,
		temp_std,
		wind_mean,
		wind_std,
		indices,
	):
		self.dates = dates
		self.no2_day = no2_day
		self.no2_mask = no2_mask
		self.temp_seq = temp_seq
		self.wind_seq = wind_seq
		self.next_no2 = next_no2
		self.no2_mean = no2_mean
		self.no2_std = no2_std
		self.temp_mean = temp_mean
		self.temp_std = temp_std
		self.wind_mean = wind_mean
		self.wind_std = wind_std
		self.indices = np.asarray(indices, dtype=np.int64)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx):
		sample_idx = int(self.indices[idx])
		no2_day = self.no2_day[sample_idx]
		no2_mask = self.no2_mask[sample_idx]
		temp_seq = self.temp_seq[sample_idx]
		wind_seq = self.wind_seq[sample_idx]
		raw_target = self.next_no2[sample_idx]

		target_mask = (~np.isnan(raw_target)).astype(np.float32)
		target = np.nan_to_num(raw_target, nan=0.0).astype(np.float32)

		no2_day = np.nan_to_num(no2_day, nan=0.0).astype(np.float32)
		no2_mask = np.nan_to_num(no2_mask, nan=0.0).astype(np.float32)
		temp_seq = np.nan_to_num(temp_seq, nan=0.0).astype(np.float32)
		wind_seq = np.nan_to_num(wind_seq, nan=0.0).astype(np.float32)

		no2_day = (no2_day - self.no2_mean) / self.no2_std
		target = (target - self.no2_mean) / self.no2_std
		temp_seq = (temp_seq - self.temp_mean) / self.temp_std
		wind_seq = (wind_seq - self.wind_mean.reshape(1, -1, 1, 1)) / self.wind_std.reshape(1, -1, 1, 1)

		x = {
			"no2_day": torch.from_numpy(no2_day),
			"no2_mask": torch.from_numpy(no2_mask),
			"temp_seq": torch.from_numpy(temp_seq),
			"wind_seq": torch.from_numpy(wind_seq),
		}
		y = {
			"next_no2": torch.from_numpy(target),
			"next_no2_mask": torch.from_numpy(target_mask),
		}
		return x, y


all_indices = np.arange(num_samples)
train_indices = all_indices[:train_end]
val_indices = all_indices[train_end:val_end]
test_indices = all_indices[val_end:]

batch_size = 4

train_dataset = WeatherAwareNO2Dataset(
	dates,
	no2_day,
	no2_mask,
	temp_seq,
	wind_seq,
	next_no2,
	no2_mean,
	no2_std,
	temp_mean,
	temp_std,
	wind_mean,
	wind_std,
	train_indices,
)
val_dataset = WeatherAwareNO2Dataset(
	dates,
	no2_day,
	no2_mask,
	temp_seq,
	wind_seq,
	next_no2,
	no2_mean,
	no2_std,
	temp_mean,
	temp_std,
	wind_mean,
	wind_std,
	val_indices,
)
test_dataset = WeatherAwareNO2Dataset(
	dates,
	no2_day,
	no2_mask,
	temp_seq,
	wind_seq,
	next_no2,
	no2_mean,
	no2_std,
	temp_mean,
	temp_std,
	wind_mean,
	wind_std,
	test_indices,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")
print(f"Test samples: {len(test_dataset):,}")


class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.block(x)


class DownBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.MaxPool2d(kernel_size=2),
			DoubleConv(in_channels, out_channels),
		)

	def forward(self, x):
		return self.block(x)


class UpBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x, skip_connection):
		x = self.up(x)
		height_diff = skip_connection.size(2) - x.size(2)
		width_diff = skip_connection.size(3) - x.size(3)
		x = nn.functional.pad(
			x,
			[
				width_diff // 2,
				width_diff - width_diff // 2,
				height_diff // 2,
				height_diff - height_diff // 2,
			],
		)
		x = torch.cat([skip_connection, x], dim=1)
		return self.conv(x)


class WeatherTemporalEncoder(nn.Module):
	def __init__(self, in_channels, embedding_dim=128):
		super().__init__()
		self.backbone = nn.Sequential(
			nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm3d(32),
			nn.ReLU(inplace=True),
			nn.Conv3d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm3d(64),
			nn.ReLU(inplace=True),
			nn.Conv3d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm3d(128),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool3d((1, 1, 1)),
		)
		self.projection = nn.Linear(128, embedding_dim)

	def forward(self, x):
		if x.ndim != 5:
			raise ValueError("Expected weather sequence with shape [batch, time, channels, height, width].")
		x = x.permute(0, 2, 1, 3, 4)
		features = self.backbone(x).flatten(1)
		return self.projection(features)


class WeatherAwareNO2FusionNet(nn.Module):
	def __init__(self, no2_channels=2, weather_embedding_dim=128, base_channels=32):
		super().__init__()
		self.inc = DoubleConv(no2_channels, base_channels)
		self.down1 = DownBlock(base_channels, base_channels * 2)
		self.down2 = DownBlock(base_channels * 2, base_channels * 4)
		self.down3 = DownBlock(base_channels * 4, base_channels * 8)
		self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
		self.temp_encoder = WeatherTemporalEncoder(in_channels=1, embedding_dim=weather_embedding_dim)
		self.wind_encoder = WeatherTemporalEncoder(in_channels=2, embedding_dim=weather_embedding_dim)
		self.weather_fuse = nn.Sequential(
			nn.Linear(weather_embedding_dim * 2, weather_embedding_dim * 2),
			nn.ReLU(inplace=True),
			nn.Linear(weather_embedding_dim * 2, base_channels * 16),
		)
		self.up1 = UpBlock(base_channels * 16, base_channels * 8)
		self.up2 = UpBlock(base_channels * 8, base_channels * 4)
		self.up3 = UpBlock(base_channels * 4, base_channels * 2)
		self.up4 = UpBlock(base_channels * 2, base_channels)
		self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)

	def forward(self, inputs):
		no2_day = inputs["no2_day"]
		no2_mask = inputs["no2_mask"]
		temp_seq = inputs["temp_seq"]
		wind_seq = inputs["wind_seq"]

		x = torch.cat([no2_day, no2_mask], dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		bottleneck = self.bottleneck(x4)

		temp_embedding = self.temp_encoder(temp_seq)
		wind_embedding = self.wind_encoder(wind_seq)
		weather_embedding = torch.cat([temp_embedding, wind_embedding], dim=1)
		weather_bias = self.weather_fuse(weather_embedding).unsqueeze(-1).unsqueeze(-1)
		bottleneck = bottleneck + weather_bias

		x = self.up1(bottleneck, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		return self.outc(x)


def masked_mse(prediction, target, target_mask, eps=1e-6):
	squared_error = (prediction - target) ** 2 * target_mask
	return squared_error.sum() / target_mask.sum().clamp_min(eps)


def run_epoch(model, loader, optimizer=None):
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_masked_mae = 0.0
	total_masked_points = 0.0

	for inputs, targets in loader:
		inputs = {key: value.to(device) for key, value in inputs.items()}
		target = targets["next_no2"].to(device)
		target_mask = targets["next_no2_mask"].to(device)

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		prediction = model(inputs)
		loss = masked_mse(prediction, target, target_mask)

		if is_train:
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

		with torch.no_grad():
			masked_abs_error = (prediction - target).abs() * target_mask
			batch_masked_points = target_mask.sum().item()
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
		for inputs, target_dict in loader:
			inputs = {key: value.to(device) for key, value in inputs.items()}
			target = target_dict["next_no2"].to(device)
			target_mask = target_dict["next_no2_mask"].to(device)
			prediction = model(inputs)

			preds.append(prediction.cpu().numpy())
			targets.append(target.cpu().numpy())
			masks.append(target_mask.cpu().numpy())

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

model = WeatherAwareNO2FusionNet().to(device)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

checkpoint_dir = Path(f"/storage/anwesha.ghosh_ug2023/weather_aware/{time.strftime('%Y%m%d_%H%M%S')}_checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_checkpoint_path = checkpoint_dir / "weather_aware_best.pt"
last_checkpoint_path = checkpoint_dir / "weather_aware_last.pt"

best_val_loss = float("inf")
best_state = None
history = []
num_epochs = 100

if last_checkpoint_path.exists():
	checkpoint = torch.load(last_checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
	best_val_loss = checkpoint.get("best_val_loss", float("inf"))
	history = checkpoint.get("history", [])
	print(f"Resumed training from {last_checkpoint_path}")

for epoch in range(1, num_epochs + 1):
	train_loss, train_mae = run_epoch(model, train_loader, optimizer=optimizer)
	val_loss, val_mae = run_epoch(model, val_loader)
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
				"no2_mean": no2_mean,
				"no2_std": no2_std,
				"temp_mean": temp_mean,
				"temp_std": temp_std,
				"wind_mean": wind_mean,
				"wind_std": wind_std,
				"batch_size": batch_size,
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
			"no2_mean": no2_mean,
			"no2_std": no2_std,
			"temp_mean": temp_mean,
			"temp_std": temp_std,
			"wind_mean": wind_mean,
			"wind_std": wind_std,
			"batch_size": batch_size,
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
		"no2_mean": no2_mean,
		"no2_std": no2_std,
		"temp_mean": temp_mean,
		"temp_std": temp_std,
		"wind_mean": wind_mean,
		"wind_std": wind_std,
		"batch_size": batch_size,
	},
	best_checkpoint_path,
)

preds, targets, masks, test_mae, test_rmse = evaluate_on_loader(model, test_loader, no2_mean, no2_std)
print(f"Weather-aware Test MAE:  {test_mae:.4f}")
print(f"Weather-aware Test RMSE: {test_rmse:.4f}")

history_df = pd.DataFrame(history)

if len(test_dataset) > 0:
	num_samples = min(8, len(test_dataset))
	indices = torch.randperm(len(test_dataset))[:num_samples].tolist()

	fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9), constrained_layout=True)
	if num_samples == 1:
		axes = np.expand_dims(axes, axis=1)

	for col, idx in enumerate(indices):
		inputs, targets_dict = test_dataset[idx]
		inputs = {key: value.unsqueeze(0).to(device) for key, value in inputs.items()}
		with torch.no_grad():
			sample_pred = model(inputs).cpu().squeeze(0).squeeze(0)

		sample_target = targets_dict["next_no2"].squeeze(0)
		sample_mask_2d = targets_dict["next_no2_mask"].squeeze(0)

		sample_target = sample_target * no2_std + no2_mean
		sample_prediction = sample_pred * no2_std + no2_mean

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
	plt.title("Weather-aware training curve")
	plt.legend()
	plt.grid(alpha=0.3)
	training_curve_path = checkpoint_dir / "training_curve_epoch_loss.png"
	plt.savefig(training_curve_path)
	plt.close()
