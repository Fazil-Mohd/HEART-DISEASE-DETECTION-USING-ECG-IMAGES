"""
download_model.py
-----------------
Downloads the trained ResNet50 ECG model from Hugging Face Hub at server
startup if the model file is not already present on disk.

Environment variables (set in Render dashboard):
    HF_MODEL_REPO      – e.g.  "your-hf-username/ecg-resnet50"
    HF_MODEL_FILENAME  – e.g.  "resnet50_ecg_best.keras"  (default)
    HF_TOKEN           – optional; only needed for private repos

The file is saved to:
    <project_root>/resnet_models/resnet50_ecg_best.keras
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_model_if_needed():
    """
    Check whether the model file exists; if not, download it from HF Hub.
    Safe to call multiple times — it is a no-op when the file is present.
    """
    # ── Resolve the expected model path ──────────────────────────────────────
    base_dir   = Path(__file__).resolve().parent
    model_dir  = base_dir / "resnet_models"
    model_path = model_dir / "resnet50_ecg_best.keras"

    if model_path.exists():
        logger.info(f"[download_model] Model already present at {model_path}")
        return  # nothing to do

    # ── Read configuration from environment ──────────────────────────────────
    hf_repo     = os.environ.get("HF_MODEL_REPO", "").strip()
    hf_filename = os.environ.get("HF_MODEL_FILENAME", "resnet50_ecg_best.keras").strip()
    hf_token    = os.environ.get("HF_TOKEN", None)  # None = public repo (no auth)

    if not hf_repo:
        logger.warning(
            "[download_model] HF_MODEL_REPO env var is not set. "
            "Skipping model download — app will use dummy predictions."
        )
        return

    # ── Download ─────────────────────────────────────────────────────────────
    try:
        from huggingface_hub import hf_hub_download

        logger.info(
            f"[download_model] Downloading '{hf_filename}' "
            f"from '{hf_repo}' → {model_path} …"
        )
        print(
            f"[download_model] Downloading model from Hugging Face Hub "
            f"({hf_repo}/{hf_filename}) …",
            flush=True,
        )

        model_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id   = hf_repo,
            filename  = hf_filename,
            token     = hf_token,
            local_dir = str(model_dir),
            local_dir_use_symlinks = False,  # copy the actual file, no symlinks
        )

        logger.info(f"[download_model] Model downloaded to {downloaded_path}")
        print(f"[download_model] ✅ Model ready at {downloaded_path}", flush=True)

    except Exception as exc:
        logger.error(f"[download_model] Failed to download model: {exc}")
        print(
            f"[download_model] ⚠️  Could not download model: {exc}\n"
            f"  App will fall back to dummy predictions.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    # Allow running directly:  python download_model.py
    logging.basicConfig(level=logging.INFO)
    download_model_if_needed()
