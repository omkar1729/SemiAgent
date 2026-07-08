"""Fine-tune google/vit-base-patch16-224 on MixedWM38 (multi-label, 38 classes).

Only needed when USE_TRAINED_VIT=true. Produces models/checkpoints/vit_best.pt.
Wafer maps are rendered with the same converter used at inference time so train
and serve see identical images.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# mlflow + transformers are imported inside main(): on Windows the DataLoader spawns
# re-import this module, so keeping heavy libs out of the top level keeps workers light.
from agents.detection_agent import wafer_array_to_pil

NUM_LABELS = 38
EPOCHS = settings.vit_epochs
BATCH_SIZE = settings.vit_batch_size
NUM_WORKERS = int(os.environ.get("VIT_NUM_WORKERS", "4"))  # parallel data loading
LR = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3

_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
TRAIN_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    _NORM,
])
VAL_TFM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _NORM,
])


class WaferMapDataset(Dataset):
    def __init__(self, data_array, labels_array, transform):
        self.data = data_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil = wafer_array_to_pil(self.data[idx])  # 3-level RGB, handles {0,1,2,3}
        image = self.transform(pil)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    use_amp = scaler is not None and device == "cuda"
    model.train() if train else model.eval()
    total_loss, exact_match, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(pixel_values=images).logits
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            exact_match += (preds == labels).all(dim=1).sum().item()
            n += images.size(0)
    return total_loss / n, exact_match / n


def _maybe_subset(indices, limit):
    """Deterministically cap an index array to `limit` (0 = no cap)."""
    if limit and limit < len(indices):
        rng = np.random.default_rng(settings.random_seed)
        return rng.choice(indices, size=limit, replace=False)
    return indices


def main():
    import mlflow
    from transformers import ViTForImageClassification

    device = settings.device
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    train_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "train_indices.npy"))
    val_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "val_indices.npy"))
    train_idx = _maybe_subset(train_idx, settings.vit_max_train_samples)
    val_idx = _maybe_subset(val_idx, settings.vit_max_val_samples)
    print(f"device={device}  train={len(train_idx)}  val={len(val_idx)}  "
          f"batch={BATCH_SIZE}  epochs={EPOCHS}  num_workers={NUM_WORKERS}")

    pin = device == "cuda"
    train_ds = WaferMapDataset(data[train_idx], labels[train_idx], TRAIN_TFM)
    val_ds = WaferMapDataset(data[val_idx], labels[val_idx], VAL_TFM)
    persist = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=persist)

    model = ViTForImageClassification.from_pretrained(
        settings.vit_model_name, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    os.makedirs(os.path.dirname(settings.vit_checkpoint_path), exist_ok=True)

    best_val, epochs_no_improve = float("inf"), 0
    with mlflow.start_run(run_name="vit_train"):
        mlflow.log_params({
            "model": settings.vit_model_name, "num_labels": NUM_LABELS,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "patience": PATIENCE,
            "device": device, "train_samples": len(train_idx),
        })
        for epoch in range(EPOCHS):
            train_loss, _ = _run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"epoch {epoch + 1}/{EPOCHS}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), settings.vit_checkpoint_path)
                print(f"  saved best checkpoint -> {settings.vit_checkpoint_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    print(f"Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
