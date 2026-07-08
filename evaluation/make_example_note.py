"""End-to-end example WITH an engineer note (counterpart to fig12, same wafer).
Run: python evaluation/make_example_note.py  ->  figures/fig16_example_note.png
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
from matplotlib.colors import ListedColormap

from agents.detection_agent import CLASS_NAMES, DetectionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.diagnosis_agent import DiagnosisAgent

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 11, "axes.titleweight": "bold"})

NOTE = ("Engineer note: a complete ring of failing dies sits at the outer wafer edge, "
        "observed after the CMP step.")

data = np.load(os.path.join(settings.mixedwm38_data_dir, "data.npy"))
labels = np.load(os.path.join(settings.mixedwm38_data_dir, "labels.npy"))
test_idx = np.load(os.path.join(settings.mixedwm38_data_dir, "test_indices.npy"))
primary = labels.argmax(1)

det = DetectionAgent(mode="vit", load_vit_checkpoint=True)
retr = RetrievalAgent()
gpt = DiagnosisAgent(provider="openai")

idx = int(test_idx[primary[test_idx] == 4][0])  # same Edge-Ring wafer as fig12
wafer = data[idx]
d = det.detect(wafer)
r = retr.retrieve(d, NOTE)          # note folded into the retrieval query
diag = gpt.diagnose(d, r, NOTE)     # note passed to the diagnosis LLM

fig = plt.figure(figsize=(13, 6.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.5], wspace=0.15)

# panel 1: wafer + engineer note
ax0 = fig.add_subplot(gs[0])
cmap = ListedColormap(["#ffffff", "#cfd8dc", "#c62828"])
ax0.imshow(np.clip(wafer, 0, 2), cmap=cmap, vmin=0, vmax=2)
ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title("1. Input wafer map + note", fontsize=12)
ax0.set_xlabel(f"True: {CLASS_NAMES[int(primary[idx])]}\n"
               f"Detected: {d['defect_type']}  (conf {d['confidence']:.2f})", fontsize=10)
ax0.text(0.5, -0.42, textwrap.fill(chr(9998) + " " + NOTE, 42), transform=ax0.transAxes,
         ha="center", va="top", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.5", fc="#fff7cc", ec="#e0b000", lw=1.2))

# panel 2: retrieved evidence
ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
ax1.set_title("2. Retrieved evidence (top 3)", fontsize=12, loc="left")
y = 0.96
for i, doc in enumerate(r["retrieved_docs"][:3], 1):
    txt = textwrap.fill(f"[{i}] " + doc.get("text", "")[:230] + "...", 46)
    ax1.text(0, y, txt, va="top", ha="left", fontsize=7.6, family="monospace")
    y -= 0.045 * (txt.count("\n") + 1) + 0.03

# panel 3: report (shows note incorporated in the process-step analysis)
ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
ax2.set_title("3. Generated RCA report (note-aware)", fontsize=12, loc="left")
hyp = (diag.get("root_cause_hypotheses") or [{}])[0]
blocks = [("Defect summary", diag.get("defect_summary", "")),
          ("Process-step analysis (incorporates note)", diag.get("process_step_analysis", "")),
          (f"Top hypothesis (rank {hyp.get('rank', 1)}, {hyp.get('confidence_level', '')})",
           hyp.get("hypothesis", "")),
          ("Supporting evidence", hyp.get("supporting_evidence", ""))]
y = 0.97
for head, body in blocks:
    ax2.text(0, y, head, va="top", fontsize=8.5, fontweight="bold", color="#1f3b6e")
    y -= 0.05
    wrapped = textwrap.fill(str(body)[:300], 52)
    ax2.text(0, y, wrapped, va="top", fontsize=7.8)
    y -= 0.048 * (wrapped.count("\n") + 1) + 0.03

fig.suptitle("End-to-end example WITH engineer note: the note flows into retrieval and the report",
             fontsize=13, y=1.02)
fig.savefig(os.path.join(OUT, "fig16_example_note.png"), bbox_inches="tight")
print("wrote figures/fig16_example_note.png")
print("note-aware process-step analysis:\n ", diag.get("process_step_analysis", "")[:300])
