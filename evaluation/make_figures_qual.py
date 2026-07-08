"""Qualitative / structural thesis figures (complement the result tables):
  fig12 — end-to-end example (wafer -> detection -> retrieved evidence -> RCA report)
  fig13 — confusion matrices, CLIP vs trained ViT (38 classes)
  fig14 — t-SNE of detector features by single-defect class (CLIP vs ViT)
  fig15 — rubric-score distributions per ablation condition (violin)
Run: python evaluation/make_figures_qual.py
"""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_all_seeds
from config import settings

set_all_seeds(settings.random_seed)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from agents.detection_agent import (CLASS_NAMES, DetectionAgent, wafer_array_to_pil,
                                     wafer_array_to_clip_pil)
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "bold"})
dev = settings.device
data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
test_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "test_indices.npy"))
primary = labels.argmax(1)


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- 12. qualitative example
def fig_example():
    det = DetectionAgent(mode="vit", load_vit_checkpoint=True)
    retr = RetrievalAgent()
    gpt = DiagnosisAgent(provider="openai")
    idx = int(test_idx[primary[test_idx] == 4][0])  # an Edge-Ring wafer (clear pattern)
    wafer = data[idx]
    d = det.detect(wafer)
    r = retr.retrieve(d, None)
    diag = gpt.diagnose(d, r, None)

    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.5], wspace=0.15)
    # panel 1: wafer
    ax0 = fig.add_subplot(gs[0])
    cmap = ListedColormap(["#ffffff", "#cfd8dc", "#c62828"])
    ax0.imshow(np.clip(wafer, 0, 2), cmap=cmap, vmin=0, vmax=2)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_title("1. Input wafer map", fontsize=12)
    ax0.set_xlabel(f"True: {CLASS_NAMES[int(primary[idx])]}\n"
                   f"Detected: {d['defect_type']}  (conf {d['confidence']:.2f})", fontsize=10)
    # panel 2: retrieved evidence
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    ax1.set_title("2. Retrieved evidence (top 3)", fontsize=12, loc="left")
    y = 0.96
    for i, doc in enumerate(r["retrieved_docs"][:3], 1):
        txt = textwrap.fill(f"[{i}] " + doc.get("text", "")[:230] + "...", 46)
        ax1.text(0, y, txt, va="top", ha="left", fontsize=7.6, family="monospace")
        y -= 0.045 * (txt.count("\n") + 1) + 0.03
    # panel 3: generated report
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    ax2.set_title("3. Generated RCA report", fontsize=12, loc="left")
    hyp = (diag.get("root_cause_hypotheses") or [{}])[0]
    blocks = [("Defect summary", diag.get("defect_summary", "")),
              (f"Top hypothesis (rank {hyp.get('rank', 1)}, {hyp.get('confidence_level', '')})",
               hyp.get("hypothesis", "")),
              ("Supporting evidence", hyp.get("supporting_evidence", "")),
              ("Investigate", "; ".join((diag.get("investigation_parameters") or [])[:3]))]
    y = 0.97
    for head, body in blocks:
        ax2.text(0, y, head, va="top", fontsize=8.5, fontweight="bold", color="#1f3b6e")
        y -= 0.05
        wrapped = textwrap.fill(str(body)[:320], 52)
        ax2.text(0, y, wrapped, va="top", fontsize=7.8)
        y -= 0.05 * (wrapped.count("\n") + 1) + 0.03
    fig.suptitle("End-to-end example: wafer image to grounded root-cause report",
                 fontsize=13, y=1.02)
    save(fig, "fig12_example.png")


# ---------------------------------------------------------------- features + preds
def extract(n=1000):
    rng = np.random.default_rng(settings.random_seed)
    samp = rng.choice(test_idx, size=min(n, len(test_idx)), replace=False)
    clip = DetectionAgent(mode="clip")
    vit = DetectionAgent(mode="vit", load_vit_checkpoint=True)
    out = {"true": [], "clip_pred": [], "vit_pred": [], "clip_f": [], "vit_f": []}
    for i in samp:
        w = data[i]
        out["true"].append(int(primary[i]))
        # CLIP
        img = clip.preprocess(wafer_array_to_clip_pil(w)).unsqueeze(0).to(dev)
        with torch.no_grad():
            cf = clip.model.encode_image(img)
            sims = (F.normalize(cf, dim=-1) @ clip.text_features.T).squeeze(0)
        out["clip_pred"].append(int(sims.argmax())); out["clip_f"].append(cf.squeeze(0).cpu().numpy())
        # ViT
        inp = vit.processor(images=wafer_array_to_pil(w), return_tensors="pt").to(dev)
        with torch.no_grad():
            o = vit.vit_model(**inp, output_hidden_states=True)
        out["vit_pred"].append(int(torch.sigmoid(o.logits).argmax()))
        out["vit_f"].append(o.hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy())
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- 13. confusion matrices
def fig_confusion(ex):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, key, name in [(axes[0], "clip_pred", "CLIP zero-shot"),
                          (axes[1], "vit_pred", "Trained ViT")]:
        cm = confusion_matrix(ex["true"], ex[key], labels=range(38), normalize="true")
        im = ax.imshow(cm, cmap="magma", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{name}", fontsize=12)
        ax.set_xlabel("Predicted class (0-37)"); ax.set_ylabel("True class (0-37)")
        ax.set_xticks([0, 8, 20, 37]); ax.set_yticks([0, 8, 20, 37])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized frequency")
    fig.suptitle("Detection confusion matrices: CLIP collapses to one class; ViT is diagonal",
                 fontsize=13)
    save(fig, "fig13_confusion.png", tight=False)


# ---------------------------------------------------------------- 14. t-SNE
def fig_tsne(ex):
    mask = (ex["true"] >= 1) & (ex["true"] <= 8)  # single-defect classes
    y = ex["true"][mask]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    cmap = plt.get_cmap("tab10")
    for ax, key, name in [(axes[0], "clip_f", "CLIP features"),
                          (axes[1], "vit_f", "Trained ViT features")]:
        Z = TSNE(n_components=2, perplexity=25, init="pca",
                 random_state=settings.random_seed).fit_transform(ex[key][mask])
        for c in range(1, 9):
            m = y == c
            ax.scatter(Z[m, 0], Z[m, 1], s=18, color=cmap((c - 1) % 10),
                       label=CLASS_NAMES[c], alpha=0.8, edgecolors="none")
        ax.set_title(name, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.suptitle("t-SNE of detector features (single-defect classes): training yields clean clusters",
                 fontsize=13)
    save(fig, "fig14_tsne.png")


# ---------------------------------------------------------------- 15. score distributions
def fig_violin():
    df = pd.read_csv("evaluation/results/study_per_image_n50.csv")
    order = [("Full (B0)", "score_grid|ViT_trained|nonote"), ("Oracle", "score_knock|oracle"),
             ("GPT-trunc", "score_knock|gpt_trunc"), ("SemiKong", "score_llm|semikong"),
             ("Sparse", "score_knock|sparse"), ("No-retr", "score_knock|noretr"),
             ("Template", "score_knock|template")]
    vals = [df[c].values for _, c in order]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    parts = ax.violinplot(vals, showmeans=True, showextrema=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#1f77b4"); pc.set_alpha(0.55)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([n for n, _ in order], rotation=20, ha="right")
    ax.set_ylabel("Rubric score / 10"); ax.set_ylim(4, 10.5); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Diagnosis score distributions by ablation condition (N=50)")
    save(fig, "fig15_score_distributions.png")


if __name__ == "__main__":
    print(f"Writing qualitative figures to {OUT}/ ...")
    try:
        fig_example()
    except Exception as e:
        print("  fig12 (example) skipped:", type(e).__name__, str(e)[:120])
    ex = extract(1000)
    fig_confusion(ex)
    fig_tsne(ex)
    fig_violin()
    print("Done.")
