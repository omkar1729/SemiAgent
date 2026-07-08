"""Evaluate a fine-tuned ViT checkpoint on the MixedWM38 test split.

Reports exact-match accuracy, per-label binary accuracy, macro-F1, and Hamming
loss; logs metrics to MLflow. Run after models/train_vit.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from agents.detection_agent import CLASS_NAMES
from models.train_vit import WaferMapDataset, VAL_TFM, NUM_LABELS


def main():
    device = settings.device
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    test_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "test_indices.npy"))

    if not os.path.exists(settings.vit_checkpoint_path):
        raise FileNotFoundError(
            f"ViT checkpoint not found at {settings.vit_checkpoint_path}. "
            "Run models/train_vit.py first.")

    test_ds = WaferMapDataset(data[test_idx], labels[test_idx], VAL_TFM)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = ViTForImageClassification.from_pretrained(
        settings.vit_model_name, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(settings.vit_checkpoint_path, map_location=device))
    model.eval().to(device)

    all_preds, all_true = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            logits = model(pixel_values=images.to(device)).logits
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_true.append(lbls.int().numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_true)

    exact_match = float((y_pred == y_true).all(axis=1).mean())
    per_label_acc = (y_pred == y_true).mean(axis=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    hl = float(hamming_loss(y_true, y_pred))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    with mlflow.start_run(run_name="vit_eval"):
        mlflow.log_metric("test_exact_match_accuracy", exact_match)
        mlflow.log_metric("test_macro_f1", macro_f1)
        mlflow.log_metric("test_hamming_loss", hl)
        mlflow.log_metric("test_mean_per_label_accuracy", float(per_label_acc.mean()))

    print(f"\nTest samples: {len(y_true)}")
    print(f"Exact-match accuracy: {exact_match:.4f}")
    print(f"Macro-F1:             {macro_f1:.4f}")
    print(f"Hamming loss:         {hl:.4f}")
    print("\nPer-label binary accuracy:")
    for i, acc in enumerate(per_label_acc):
        print(f"  {i:2d} {CLASS_NAMES[i]:32s} {acc:.4f}")


if __name__ == "__main__":
    main()
