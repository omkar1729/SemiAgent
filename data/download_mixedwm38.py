"""Download / prepare the MixedWM38 dataset.

Produces, under settings.mixedwm38_data_dir (default data/processed/):
    data.npy          (38015, 52, 52) uint8 wafer maps
    labels.npy        (38015, 38)     uint8 one-hot over the 38 combination classes
    train_indices.npy / val_indices.npy / test_indices.npy   (70/15/15 stratified)

The raw Kaggle file stores labels as an (N, 8) multi-hot of the base defect types;
we convert that to the 38-class one-hot used everywhere downstream (see
agents.detection_agent.onehot38_from_multihot8).
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split

from utils.seed import set_all_seeds
from config import settings
from agents.detection_agent import CLASS_NAMES, onehot38_from_multihot8

set_all_seeds(settings.random_seed)


def _locate_npz() -> str:
    """Find the MixedWM38 .npz, downloading from Kaggle via opendatasets if absent."""
    search_dirs = [settings.mixedwm38_data_dir, "data", "."]
    for d in search_dirs:
        hits = glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)
        if hits:
            return hits[0]

    print("No local .npz found — downloading from Kaggle (prompts for username + key once)...")
    import opendatasets as od

    od.download("https://www.kaggle.com/datasets/co1emi11er2/mixedwm38-dataset")
    hits = glob.glob(os.path.join("**", "*.npz"), recursive=True)
    if not hits:
        raise FileNotFoundError("Download finished but no .npz file was found.")
    return hits[0]


def main():
    os.makedirs(settings.mixedwm38_data_dir, exist_ok=True)

    npz_path = _locate_npz()
    print(f"Loading {npz_path}")
    npz = np.load(npz_path)
    # The dataset stores arrays positionally: arr_0 = maps, arr_1 = labels.
    keys = list(npz.keys())
    maps = npz[keys[0]].astype(np.uint8)          # (N, 52, 52), values {0,1,2,3}
    labels8 = npz[keys[1]].astype(np.uint8)       # (N, 8) multi-hot base types
    print(f"maps {maps.shape} {maps.dtype} | raw labels {labels8.shape}")

    labels38 = onehot38_from_multihot8(labels8)   # (N, 38) one-hot
    primary = labels38.argmax(axis=1)             # single-class proxy for stratification

    data_path = os.path.join(settings.mixedwm38_data_dir, "data.npy")
    labels_path = os.path.join(settings.mixedwm38_data_dir, "labels.npy")
    np.save(data_path, maps)
    np.save(labels_path, labels38)
    print(f"Saved {data_path} and {labels_path}")

    # 70/15/15 stratified split on the primary class.
    idx = np.arange(len(maps))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, random_state=settings.random_seed, stratify=primary
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=settings.random_seed, stratify=primary[temp_idx]
    )
    for name, arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.save(os.path.join(settings.mixedwm38_data_dir, f"{name}_indices.npy"), arr)
    print(f"Split sizes -> train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    print("\nClass distribution (full dataset):")
    counts = labels38.sum(axis=0).astype(int)
    for i, c in enumerate(counts):
        print(f"  {i:2d} {CLASS_NAMES[i]:32s} {c}")


if __name__ == "__main__":
    main()
