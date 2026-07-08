"""Central configuration.

All modules import the shared `settings` object via `from config import settings`.
Values are loaded from the project `.env` automatically (pydantic-settings).
"""
import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# MLflow 3.x raises on the bare-directory file store unless explicitly opted in.
# We keep the simple local file-store backend (mlflow_runs) the project expects.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


class Settings(BaseSettings):
    # --- credentials ---
    openai_api_key: str = ""
    uspto_api_key: str = ""
    serpapi_api_key: str = ""  # SerpApi Google Patents (preferred patent source)
    gcp_project: str = ""  # set to enable the BigQuery (Google Patents) corpus source

    # --- diagnosis LLM backend ---
    # "openai"         -> OpenAI gpt-4o-mini
    # "semikong"       -> an OpenAI-compatible server (LMStudio/vLLM/Ollama) for SemiKong
    # "semikong_local" -> SemiKong loaded in-process via transformers + bitsandbytes 4-bit
    diagnosis_provider: str = "openai"
    semikong_base_url: str = "http://localhost:1234/v1"
    semikong_model: str = "semikong"
    semikong_api_key: str = "not-needed"
    # Which SemiKong backend the ablation's condition E uses. Default "semikong"
    # (remote OpenAI-compatible server) fails fast/cleanly when none is running;
    # set "semikong_local" to load the model in-process via transformers+bnb.
    semikong_backend: str = "semikong"
    semikong_hf_model: str = "pentagoniac/SEMIKONG-8B-chat"  # gated; needs hf_token
    hf_token: str = ""

    # --- data / artifact paths ---
    mixedwm38_data_dir: str = "data/processed"
    corpus_jsonl_path: str = "data/corpus.jsonl"
    chroma_persist_dir: str = "knowledge_base/chroma_db"
    mlflow_tracking_uri: str = "mlflow_runs"

    # --- models ---
    llm_model: str = "gpt-4o-mini"
    use_trained_vit: bool = False
    clip_model: str = "ViT-B/32"
    clip_confidence_threshold: float = 0.15
    vit_model_name: str = "google/vit-base-patch16-224"
    vit_checkpoint_path: str = "models/checkpoints/vit_best.pt"
    vit_num_labels: int = 38
    # ViT training knobs (GPU-friendly defaults; 0 = use the full split)
    vit_batch_size: int = 16
    vit_epochs: int = 20
    vit_max_train_samples: int = 0
    vit_max_val_samples: int = 0

    # --- retrieval ---
    retrieval_top_k: int = 5
    rrf_k_constant: int = 60

    # --- corpus ---
    corpus_max_docs: int = 1000

    # --- misc ---
    random_seed: int = 42
    device: str = "auto"  # "auto" resolves to cuda when available, else cpu

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# Resolve "auto" device to the best available backend.
if settings.device == "auto":
    try:
        import torch

        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        settings.device = "cpu"
