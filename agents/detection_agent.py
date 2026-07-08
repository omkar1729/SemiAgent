"""Detection Agent — two modes.

Mode A (default): CLIP zero-shot. The 38 MixedWM38 class descriptions are encoded
once and every wafer image is matched against them — no training required.
Mode B (optional): a ViT fine-tuned on MixedWM38, enabled with USE_TRAINED_VIT=true.

This module is also the single source of truth for the class taxonomy
(`CLASS_NAMES`, `CLIP_DESCRIPTIONS`) and for mapping the raw MixedWM38 8-bit
multi-hot labels onto the 38 combination classes. Heavy ML libraries are imported
lazily inside methods so other scripts can import the taxonomy cheaply.
"""
from typing import Union

import numpy as np
from PIL import Image

from config import settings

# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------
# 38 MixedWM38 classes: index 0 = Normal, 1-8 = single defect types,
# 9-37 = the 29 mixed combinations (verified against the dataset to cover every
# one of the 38 unique label patterns present).
CLASS_NAMES = [
    "Normal",
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Near-Full", "Random", "Scratch",
    "Center+Edge-Loc", "Center+Edge-Ring", "Center+Local", "Center+Scratch",
    "Donut+Edge-Loc", "Donut+Edge-Ring", "Donut+Local", "Donut+Scratch",
    "Edge-Loc+Local", "Edge-Loc+Scratch", "Edge-Ring+Local", "Edge-Ring+Scratch",
    "Local+Scratch",
    "Center+Edge-Loc+Local", "Center+Edge-Loc+Scratch", "Center+Edge-Ring+Local",
    "Center+Edge-Ring+Scratch", "Center+Local+Scratch",
    "Donut+Edge-Loc+Local", "Donut+Edge-Loc+Scratch", "Donut+Edge-Ring+Local",
    "Donut+Edge-Ring+Scratch", "Donut+Local+Scratch",
    "Edge-Loc+Local+Scratch", "Edge-Ring+Local+Scratch",
    "Center+Edge-Loc+Local+Scratch", "Center+Edge-Ring+Local+Scratch",
    "Donut+Edge-Loc+Local+Scratch", "Donut+Edge-Ring+Local+Scratch",
]
assert len(CLASS_NAMES) == 38

# Order of the 8 columns in the raw MixedWM38 label array (arr_1). This is the
# dataset authors' official mapping (Junliangwangdhu/WaferMap): column i is set iff
# base type DATA_COLUMN_TYPES[i] is present. Verified against the published
# single-type index ranges (e.g. rows 25000-25999 -> [0,0,1,0,0,0,0,0] = Edge-Loc,
# rows 37015-38014 -> [0,0,0,0,0,0,1,0] = Scratch). "Loc" is named "Local" here to
# match CLASS_NAMES. NOTE: Random appears only as a single type, never in a mix.
DATA_COLUMN_TYPES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Local", "Near-Full", "Scratch", "Random",
]

# Single-type descriptions exactly as specified.
_BASE_DESCRIPTIONS = {
    "Normal": "a semiconductor wafer map with no defects, uniform and clean across the entire surface",
    "Center": "a semiconductor wafer map with a circular cluster of defects concentrated at the center",
    "Donut": "a semiconductor wafer map with a ring-shaped defect pattern surrounding the center area",
    "Edge-Loc": "a semiconductor wafer map with defects localized along one edge of the wafer",
    "Edge-Ring": "a semiconductor wafer map with defects forming a complete ring around the outer edge",
    "Local": "a semiconductor wafer map with a small localized cluster of defects in one region",
    "Near-Full": "a semiconductor wafer map with defects covering nearly the entire wafer surface",
    "Random": "a semiconductor wafer map with randomly scattered defects distributed across the surface",
    "Scratch": "a semiconductor wafer map with a linear scratch pattern of defects crossing the surface",
}
# Core phrase (drop the shared "a semiconductor wafer map with " prefix) used to
# compose descriptions for mixed classes.
_PREFIX = "a semiconductor wafer map with "


def _build_descriptions() -> dict:
    out = {}
    for name in CLASS_NAMES:
        if name in _BASE_DESCRIPTIONS:
            out[name] = _BASE_DESCRIPTIONS[name]
        else:  # mixed: combine the component descriptions
            parts = [_BASE_DESCRIPTIONS[t][len(_PREFIX):] for t in name.split("+")]
            out[name] = _PREFIX + ", and also ".join(parts)
    return out


CLIP_DESCRIPTIONS = _build_descriptions()

# Map a set of active base types -> 38-class index.
_CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
_TYPESET_TO_INDEX = {
    frozenset(name.split("+")): i for i, name in enumerate(CLASS_NAMES) if name != "Normal"
}


def class_index_from_multihot8(row) -> int:
    """Map one raw MixedWM38 8-bit multi-hot label row to its 38-class index."""
    active = frozenset(DATA_COLUMN_TYPES[i] for i in range(8) if row[i] == 1)
    if not active:
        return 0  # Normal
    return _TYPESET_TO_INDEX[active]


def onehot38_from_multihot8(labels8: np.ndarray) -> np.ndarray:
    """Convert an (N, 8) multi-hot array to an (N, 38) one-hot array."""
    labels8 = np.asarray(labels8)
    out = np.zeros((labels8.shape[0], 38), dtype=np.uint8)
    for i, row in enumerate(labels8):
        out[i, class_index_from_multihot8(row)] = 1
    return out


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------
def wafer_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a (52, 52) wafer map to a 3-level RGB PIL image (general / ViT use).

    Handles both the binary {0,1} layout assumed by the brief and the real
    MixedWM38 layout {0=background, 1=good die, 2/3=defect die}. Defects render
    bright on a faint wafer disk so spatial patterns and the wafer outline are
    both visible.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if int(arr.max()) <= 1:
        img = (arr * 255).astype(np.uint8)
    else:
        img = np.zeros(arr.shape, dtype=np.uint8)
        img[arr == 1] = 80     # good die -> faint wafer disk
        img[arr >= 2] = 255    # defect die -> bright
    return Image.fromarray(img, mode="L").convert("RGB")


def wafer_array_to_clip_pil(arr: np.ndarray) -> Image.Image:
    """Render a wafer map for CLIP: dark defects on a white field, crisply upscaled.

    Empirically this high-contrast rendering separates the gross spatial patterns
    (center / donut / random) better than a faint-disk rendering, which CLIP tends
    to collapse onto a single class. Nearest-neighbour upscaling keeps the blocky
    defect structure sharp before CLIP's own bicubic resize blurs it.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if int(arr.max()) <= 1:
        img = np.where(arr >= 1, 0, 255).astype(np.uint8)
    else:
        img = np.full(arr.shape, 255, dtype=np.uint8)
        img[arr >= 2] = 0      # defect die -> black on white
    pil = Image.fromarray(img, mode="L").convert("RGB")
    return pil.resize((224, 224), Image.NEAREST)


# Light prompt ensemble: the spec description plus two rephrasings, averaged.
_CLIP_TEMPLATES = [
    "{d}",
    "a black and white wafer bin map, {d}",
    "a semiconductor defect map showing {d}",
]


# ---------------------------------------------------------------------------
# Detection Agent
# ---------------------------------------------------------------------------
class DetectionAgent:
    def __init__(self, mode: str = None, load_vit_checkpoint: bool = True):
        """mode: None -> from settings (clip/vit); or force "clip"/"vit".
        load_vit_checkpoint=False loads the base ViT with a random 38-class head
        (an *untrained* detector) for ablation."""
        if mode is None:
            mode = "vit" if settings.use_trained_vit else "clip"
        if mode == "vit":
            self._init_vit(load_checkpoint=load_vit_checkpoint)
        else:
            self._init_clip()

    # --- CLIP zero-shot mode ---
    def _init_clip(self):
        import torch
        import torch.nn.functional as F

        self._torch = torch
        try:  # prefer the openai/CLIP package if installed (matches the brief)
            import clip

            self.model, self.preprocess = clip.load(settings.clip_model, device=settings.device)
            self._tokenize = clip.tokenize
            self.clip_backend = "openai-clip"
        except ImportError:  # fall back to open_clip with the same OpenAI weights
            import open_clip

            arch = settings.clip_model.replace("/", "-")  # "ViT-B/32" -> "ViT-B-32"
            # OpenAI CLIP weights were trained with the QuickGELU activation;
            # force it so the pretrained weights load into a matching graph.
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                arch, pretrained="openai", device=settings.device, force_quick_gelu=True
            )
            self.model.eval()
            self._tokenize = open_clip.get_tokenizer(arch)
            self.clip_backend = "open-clip"

        # Ensembled text features: average each class's templated prompts.
        feats = []
        for name in CLASS_NAMES:
            prompts = [t.format(d=CLIP_DESCRIPTIONS[name]) for t in _CLIP_TEMPLATES]
            tokens = self._tokenize(prompts).to(settings.device)
            with torch.no_grad():
                emb = F.normalize(self.model.encode_text(tokens), dim=-1).mean(dim=0)
                emb = F.normalize(emb, dim=0)
            feats.append(emb)
        self.text_features = torch.stack(feats)
        self.mode = "clip"
        self.detection_mode_label = "clip"

    # --- ViT mode (trained checkpoint, or untrained base for ablation) ---
    def _init_vit(self, load_checkpoint: bool = True):
        import os

        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        self._torch = torch
        self.processor = ViTImageProcessor.from_pretrained(settings.vit_model_name)
        self.vit_model = ViTForImageClassification.from_pretrained(
            settings.vit_model_name,
            num_labels=settings.vit_num_labels,
            ignore_mismatched_sizes=True,
        )
        if load_checkpoint:
            if not os.path.exists(settings.vit_checkpoint_path):
                raise FileNotFoundError(
                    f"ViT checkpoint not found at {settings.vit_checkpoint_path}. "
                    "Run models/train_vit.py first or set USE_TRAINED_VIT=false in .env"
                )
            state_dict = torch.load(settings.vit_checkpoint_path, map_location=settings.device)
            self.vit_model.load_state_dict(state_dict)
            self.detection_mode_label = "vit"
        else:
            # base ViT with a randomly-initialized classification head: untrained.
            self.detection_mode_label = "vit_untrained"
        self.vit_model.eval().to(settings.device)
        self.mode = "vit"

    # --- shared image loading ---
    @staticmethod
    def _load_raw(image_input: Union[str, np.ndarray]):
        """Return ("array", wafer_map) for raw maps, else ("pil", image)."""
        if isinstance(image_input, Image.Image):
            return "pil", image_input.convert("RGB")
        if isinstance(image_input, np.ndarray):
            return "array", image_input
        if isinstance(image_input, str):
            if image_input.lower().endswith(".npy"):
                return "array", np.load(image_input)
            return "pil", Image.open(image_input).convert("RGB")
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def detect(self, image_input: Union[str, np.ndarray]) -> dict:
        kind, obj = self._load_raw(image_input)

        if self.mode == "clip":
            pil_image = wafer_array_to_clip_pil(obj) if kind == "array" else obj
            probabilities = self._detect_clip(pil_image)
        else:
            pil_image = wafer_array_to_pil(obj) if kind == "array" else obj
            probabilities = self._detect_vit(pil_image)

        probabilities = np.asarray(probabilities, dtype=float).ravel()
        argmax = int(probabilities.argmax())
        threshold = settings.clip_confidence_threshold
        return {
            "defect_type": CLASS_NAMES[argmax],
            "confidence": float(probabilities[argmax]),
            "active_defects": [
                CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if probabilities[i] > threshold
            ],
            "all_probabilities": {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))},
            "detection_mode": self.detection_mode_label,
        }

    def _detect_clip(self, pil_image: Image.Image) -> np.ndarray:
        torch = self._torch
        import torch.nn.functional as F

        image = self.preprocess(pil_image).unsqueeze(0).to(settings.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            similarities = (image_features @ self.text_features.T).squeeze(0)
            probs = torch.softmax(similarities / 0.07, dim=-1)
        return probs.cpu().numpy()

    def _detect_vit(self, pil_image: Image.Image) -> np.ndarray:
        torch = self._torch
        inputs = self.processor(images=pil_image, return_tensors="pt").to(settings.device)
        with torch.no_grad():
            logits = self.vit_model(**inputs).logits
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().squeeze()
