"""Intrinsic detection accuracy for all three detectors on one fixed 500-sample
test subset: CLIP zero-shot, untrained ViT (random head), and the trained ViT.

Reports overall top-1 primary-class accuracy plus per-class accuracy for the 9
single-type classes (Normal + 8 singles). This is the ONLY place detector accuracy
is measured; the diagnosis study measures the detectors' downstream effect instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import mlflow
import numpy as np
from sklearn.metrics import f1_score

from agents.detection_agent import CLASS_NAMES, DetectionAgent

N_SAMPLES = 500


def eval_one(agent, data, sample, primary):
    preds, gts = [], []
    per_correct = np.zeros(9, dtype=int)
    per_total = np.zeros(9, dtype=int)
    for i in sample:
        out = agent.detect(data[i])
        pred = int(np.argmax(list(out["all_probabilities"].values())))
        gt = int(primary[i])
        preds.append(pred)
        gts.append(gt)
        if gt < 9:
            per_total[gt] += 1
            per_correct[gt] += int(pred == gt)
    top1 = float(np.mean(np.array(preds) == np.array(gts)))
    # macro-F1 over the classes actually present in the sample (the thesis Table-4 metric)
    macro_f1 = float(f1_score(gts, preds, average="macro", zero_division=0))
    per_class = {CLASS_NAMES[c]: (per_correct[c] / per_total[c]) for c in range(9) if per_total[c]}
    return top1, macro_f1, per_class


def main():
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    test_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "test_indices.npy"))
    primary = labels.argmax(axis=1)

    rng = np.random.default_rng(settings.random_seed)
    sample = rng.choice(test_idx, size=min(N_SAMPLES, len(test_idx)), replace=False)

    modes = [("CLIP", lambda: DetectionAgent(mode="clip")),
             ("ViT_untrained", lambda: DetectionAgent(mode="vit", load_vit_checkpoint=False))]
    if os.path.exists(settings.vit_checkpoint_path):
        modes.append(("ViT_trained", lambda: DetectionAgent(mode="vit", load_vit_checkpoint=True)))
    else:
        print(f"No ViT checkpoint at {settings.vit_checkpoint_path}; skipping ViT_trained.")

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("wafer-rca")
    with mlflow.start_run(run_name="eval_detection"):
        mlflow.log_param("n_samples", len(sample))
        for name, make in modes:
            top1, macro_f1, per_class = eval_one(make(), data, sample, primary)
            mlflow.log_metric(f"{name}_top1_accuracy", top1)
            mlflow.log_metric(f"{name}_macro_f1", macro_f1)
            for cls, acc in per_class.items():
                mlflow.log_metric(f"{name}_acc_{cls}", acc)
            print(f"\n{name}  |  samples: {len(sample)}  |  top-1 = {top1:.4f}  |  macro-F1 = {macro_f1:.4f}")
            print("  per-class (single-type):  "
                  + "  ".join(f"{cls}={acc:.3f}" for cls, acc in per_class.items()))


if __name__ == "__main__":
    main()
