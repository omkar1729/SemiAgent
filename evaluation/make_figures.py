"""Generate publication-quality thesis figures from the verified results.

Outputs PNGs (300 dpi) into figures/. Pulls from the saved CSVs and MLflow where
possible; a few small verified scalars (detection, retrieval per-query, latency) are
inlined. Run:  python evaluation/make_figures.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from config import settings

R = "evaluation/results"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
                     "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"clip": "#9e9e9e", "untr": "#ffb74d", "trained": "#1f77b4", "accent": "#d62728",
     "good": "#2ca02c", "gpt": "#1f77b4", "semikong": "#d62728"}


def save(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- 1. architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.6, 3.2))
    ax.axis("off"); ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.2)
    boxes = [(0.2, "Wafer map (.npy)\n+ optional\nengineer note", "#eeeeee"),
             (2.4, "Detection Agent\n(CLIP / ViT-Base)", "#cfe8ff"),
             (4.8, "Retrieval Agent\n(BM25 + dense, RRF)", "#d7f0d7"),
             (7.2, "Diagnosis Agent\n(GPT-4o-mini, CoT)", "#ffe0cc"),
             (9.6, "Structured\nRCA report", "#eeeeee")]
    for x, label, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.1), 1.7, 1.1, boxstyle="round,pad=0.04",
                                    fc=col, ec="#333", lw=1.3))
        ax.text(x + 0.85, 1.65, label, ha="center", va="center", fontsize=9.5)
    for x in [2.0, 4.4, 6.8, 9.2]:
        ax.annotate("", xy=(x + 0.35, 1.65), xytext=(x, 1.65),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333"))
    # knowledge base feeding retrieval
    ax.add_patch(FancyBboxPatch((4.8, 0.05), 1.7, 0.7, boxstyle="round,pad=0.03",
                                fc="#f5f5dc", ec="#777", lw=1))
    ax.text(5.65, 0.4, "Knowledge base\n~1000 docs", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(5.65, 1.05), xytext=(5.65, 0.78),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#777"))
    ax.text(5.5, 2.9, "SemiAgent — three-agent wafer defect RCA pipeline (LangGraph)",
            ha="center", fontsize=12, fontweight="bold")
    save(fig, "fig1_architecture.png")


# ---------------------------------------------------------------- 2. wafer examples
def fig_wafer_examples():
    from agents.detection_agent import CLASS_NAMES
    data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"), mmap_mode="r")
    labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
    primary = labels.argmax(1)
    cmap = ListedColormap(["#ffffff", "#cfd8dc", "#c62828"])  # bg / good / defect
    fig, axes = plt.subplots(2, 4, figsize=(9, 5.6))
    for c, ax in zip(range(1, 9), axes.ravel()):
        idx = int(np.where(primary == c)[0][0])
        ax.imshow(np.clip(np.asarray(data[idx]), 0, 2), cmap=cmap, vmin=0, vmax=2)
        ax.set_title(CLASS_NAMES[c], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("MixedWM38 single-defect patterns (white=outside, grey=good die, red=failing die)",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.subplots_adjust(hspace=0.45, wspace=0.12, top=0.9, bottom=0.03)
    save(fig, "fig2_wafer_examples.png", tight=False)


# ---------------------------------------------------------------- 3. detection
def fig_detection():
    det = ["CLIP\nzero-shot", "ViT\nuntrained", "ViT\ntrained"]
    top1 = [0.050, 0.044, 0.984]; f1 = [0.008, 0.008, 0.982]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    b1 = ax.bar(x - w / 2, top1, w, label="Top-1 accuracy", color=C["trained"])
    b2 = ax.bar(x + w / 2, f1, w, label="Macro-F1", color=C["good"])
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(det); ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score"); ax.set_title("Detection accuracy (500 test maps)")
    ax.legend(frameon=False)
    save(fig, "fig3_detection_accuracy.png")


# ---------------------------------------------------------------- 4. ViT training
def fig_training():
    runs = glob.glob("mlflow_runs/*/*/metrics/val_accuracy")
    run = max(runs, key=os.path.getmtime)  # latest = corrected-label run
    def series(metric):
        path = run.replace("val_accuracy", metric)
        if not os.path.exists(path):
            return None
        rows = [ln.split() for ln in open(path)]
        rows.sort(key=lambda r: int(r[2]))
        return [float(r[1]) for r in rows]
    va, vl, tl = series("val_accuracy"), series("val_loss"), series("train_loss")
    ep = list(range(1, len(va) + 1))
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(ep, va, "o-", color=C["trained"], label="val accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Validation accuracy", color=C["trained"])
    ax1.set_ylim(0.94, 1.0); ax1.tick_params(axis="y", labelcolor=C["trained"])
    ax1.annotate(f"best {max(va):.3f}", xy=(ep[va.index(max(va))], max(va)),
                 xytext=(len(ep) - 3, 0.952), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="#555"))
    ax2 = ax1.twinx(); ax2.grid(False)
    if vl: ax2.plot(ep, vl, "s--", color=C["accent"], label="val loss")
    if tl: ax2.plot(ep, tl, "^:", color="#ff9896", label="train loss")
    ax2.set_ylabel("Loss", color=C["accent"]); ax2.tick_params(axis="y", labelcolor=C["accent"])
    ax1.set_title("ViT fine-tuning on MixedWM38")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="center right")
    save(fig, "fig4_vit_training.png")


# ---------------------------------------------------------------- 5. retrieval
def fig_retrieval():
    q = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Near-Full", "Random", "Scratch"]
    hyb = [.20, .20, .60, .60, .40, .60, .40, .60]
    spa = [.20, .40, .60, .60, .60, .60, .40, .80]
    den = [.40, .40, .60, .60, .40, .60, .20, .60]
    x = np.arange(8); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w, hyb, w, label="hybrid (mean 0.45)", color=C["trained"])
    ax.bar(x, spa, w, label="sparse/BM25 (mean 0.53)", color=C["good"])
    ax.bar(x + w, den, w, label="dense (mean 0.48)", color=C["untr"])
    ax.axhline(0.85, ls="--", color=C["accent"], lw=1, label="target 0.85")
    ax.set_xticks(x); ax.set_xticklabels(q, rotation=30, ha="right")
    ax.set_ylabel("Precision@5"); ax.set_ylim(0, 1.0)
    ax.set_title("Retrieval Precision@5 by mode (cosine relevance of hybrid = 0.671)")
    ax.legend(frameon=False, ncol=2, fontsize=8.5)
    save(fig, "fig5_retrieval.png")


# ---------------------------------------------------------------- 6. ablation N=50
def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."

def fig_ablation_bar():
    df = pd.read_csv(f"{R}/study_knockout_n50.csv")
    order = ["grid|ViT_trained|nonote", "knock|oracle", "knock|gpt_trunc", "llm|semikong",
             "knock|sparse", "knock|noretr", "knock|template"]
    lbl = {"grid|ViT_trained|nonote": "Full system (B0)", "knock|oracle": "Oracle detection",
           "knock|gpt_trunc": "GPT, truncated docs", "llm|semikong": "SemiKong writer",
           "knock|sparse": "Sparse-only retrieval", "knock|noretr": "No retrieval",
           "knock|template": "Template (no LLM)"}
    df = df.set_index("condition")
    vals = [df.loc[c, "mean_score"] for c in order]
    errs = [df.loc[c, "std_score"] for c in order]
    ps = [df.loc[c, "p_value_vs_B0"] for c in order]
    cols = [C["good"]] + [C["trained"]] * 3 + [C["untr"], C["accent"], C["accent"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b = ax.bar(range(len(order)), vals, yerr=errs, capsize=3, color=cols, ec="#333", lw=0.6)
    for i, (v, p) in enumerate(zip(vals, ps)):
        tag = "baseline" if (isinstance(p, float) and np.isnan(p)) else _stars(p)
        ax.text(i, v + errs[i] + 0.12, f"{v:.2f}\n{tag}", ha="center", fontsize=8)
    ax.axhline(vals[0], ls=":", color="#555", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([lbl[c] for c in order], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Rubric score / 10"); ax.set_ylim(0, 11)
    ax.set_title("Component knockout vs full system (N=50; *** p<.001, ** p<.01, * p<.05)")
    save(fig, "fig6_ablation_knockout.png")


# ---------------------------------------------------------------- 7. ablation across N
def fig_ablation_acrossN():
    Ns = [10, 30, 50]
    conds = [("grid|ViT_trained|nonote", "Full system", C["good"], "o-"),
             ("knock|template", "Template (no LLM)", C["accent"], "s-"),
             ("knock|noretr", "No retrieval", "#ff7f0e", "^-"),
             ("knock|sparse", "Sparse-only", C["untr"], "d--"),
             ("llm|semikong", "SemiKong", "#9467bd", "v--"),
             ("knock|oracle", "Oracle detection", C["trained"], "x:")]
    data = {N: pd.read_csv(f"{R}/study_knockout_n{N}.csv").set_index("condition") for N in Ns}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for c, label, col, st in conds:
        ys = [data[N].loc[c, "mean_score"] for N in Ns]
        ax.plot(Ns, ys, st, color=col, label=label, lw=1.8, ms=7)
    ax.set_xticks(Ns); ax.set_xlabel("Sample size N"); ax.set_ylabel("Rubric score / 10")
    ax.set_ylim(7, 10); ax.set_title("Ablation stability across sample size")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)
    save(fig, "fig7_ablation_across_n.png")


# ---------------------------------------------------------------- 8. detector x note grid
def fig_grid():
    df = pd.read_csv(f"{R}/study_grid_n50.csv")
    piv = df.pivot(index="detector", columns="note", values="mean_score")
    piv = piv.reindex(["CLIP", "ViT_untrained", "ViT_trained"])[["nonote", "note"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(piv.values, cmap="YlGnBu", vmin=9.0, vmax=10.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["no note", "with note"])
    ax.set_yticks(range(3)); ax.set_yticklabels(["CLIP", "ViT untrained", "ViT trained"])
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="#222")
    ax.grid(False); ax.set_title("Diagnosis score /10 by detector x note (N=50)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="score /10")
    save(fig, "fig8_detector_note_grid.png")


# ---------------------------------------------------------------- 9. multi-judge
def fig_multijudge():
    mj = pd.read_csv(f"{R}/multi_judge_n20.csv")
    piv = mj.pivot_table(index="judge", columns="writer", values="overall", aggfunc="mean")
    judges = ["gpt-4o-mini", "claude", "gemini", "semikong"]
    piv = piv.reindex(judges)
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(x - w / 2, piv["gpt"], w, label="GPT-4o-mini written", color=C["gpt"])
    ax.bar(x + w / 2, piv["semikong"], w, label="SemiKong-8B written", color=C["semikong"])
    for i, j in enumerate(judges):
        ax.text(i - w / 2, piv["gpt"][j] + 0.1, f"{piv['gpt'][j]:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, piv["semikong"][j] + 0.1, f"{piv['semikong'][j]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["GPT-4o-mini\n(judge)", "Claude\n(judge)",
                                          "Gemini\n(judge)", "SemiKong\n(judge)"])
    ax.set_ylabel("Mean rubric score / 10"); ax.set_ylim(0, 11)
    ax.set_title("Multi-judge cross-check: every judge ranks GPT > SemiKong (N=20)")
    ax.legend(frameon=False)
    save(fig, "fig9_multijudge.png")


# ---------------------------------------------------------------- 10. latency
def fig_latency():
    parts = [("Detection", 88), ("Retrieval", 18), ("Diagnosis (LLM)", 4451)]
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    left = 0
    cols = [C["trained"], C["good"], C["accent"]]
    for (name, ms), col in zip(parts, cols):
        ax.barh(0, ms, left=left, color=col, ec="white")
        if ms > 200:
            ax.text(left + ms / 2, 0, f"{name}\n{ms} ms ({ms/4557*100:.0f}%)",
                    ha="center", va="center", color="white", fontsize=9)
        left += ms
    ax.text(88 + 18 + 60, 0.55, "Detection 88 + Retrieval 18 ms", fontsize=8, color="#333")
    ax.set_xlim(0, 4700); ax.set_yticks([]); ax.set_xlabel("milliseconds per wafer")
    ax.set_title("End-to-end latency = 4,557 ms/case (~98% is the LLM call)")
    ax.grid(False)
    save(fig, "fig10_latency.png")


# ---------------------------------------------------------------- 11. rubric radar
def fig_rubric_radar():
    dims = ["Causal\ncorrectness", "Evidence\ngrounding", "Technical\naccuracy",
            "Completeness", "Actionability"]
    vals = [5.0, 5.0, 5.0, 4.0, 5.0]
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    vals2 = vals + vals[:1]; ang2 = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(5.2, 5), subplot_kw=dict(polar=True))
    ax.plot(ang2, vals2, "o-", color=C["trained"], lw=2)
    ax.fill(ang2, vals2, color=C["trained"], alpha=0.25)
    ax.set_xticks(ang); ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 5); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_title("Full-system rubric profile (B0, /5)")
    save(fig, "fig11_rubric_dimensions.png")


if __name__ == "__main__":
    print(f"Writing figures to {OUT}/ ...")
    fig_architecture()
    fig_wafer_examples()
    fig_detection()
    fig_training()
    fig_retrieval()
    fig_ablation_bar()
    fig_ablation_acrossN()
    fig_grid()
    fig_multijudge()
    fig_latency()
    fig_rubric_radar()
    print("Done.")
