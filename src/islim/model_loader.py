
from pathlib import Path

import torch
import gdown
from transformers import T5Tokenizer


# --------------------------------------------------
# 1. PATHS AND DEVICE
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "islim"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. PRETRAINED MODEL FILES
# --------------------------------------------------

MODEL_FILES = {
    "encoder_traced.ptl": "1xd6SCKhaGCY6K7OYpmwVR3rtRHX83tiy",
    "t5_encoder.ptl": "1QhRHsqzO_BkOZ8VLHL7XM0ViF3nMka0e",
    "t5_decoder_first_step.ptl": "1d1_LADod4HfQnRx2OD_CsTzjbRo19xhN",
    "t5_decoder_step.ptl": "1Emx6mays0_5naLZJ-nm-zxIYP1uRisSn",
    "t5_lm_head.ptl": "17x3cQjzfrdX9mvVdJvG_6jrFqAF_qEyj",
}


# --------------------------------------------------
# 3. DOWNLOAD MODEL FILES IF MISSING
# --------------------------------------------------

def download_models():

    for filename, file_id in MODEL_FILES.items():

        file_path = MODEL_DIR / filename

        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"[OK] Already downloaded: {filename}")
            continue

        print(f"[DOWNLOAD] {filename}")

        gdown.download(
            id=file_id,
            output=str(file_path),
            quiet=False
        )

        if not file_path.exists() or file_path.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {filename}")

    print("\nAll five model files are available.")


# --------------------------------------------------
# 4. LOAD TORCHSCRIPT COMPONENTS
# --------------------------------------------------


def load_torchscript(path):
    try:
        return torch.jit._load_for_lite_interpreter(str(path))
    except Exception:
        return torch.jit.load(
            str(path),
            map_location=DEVICE
        )



# --------------------------------------------------
# 5. LOAD COMPLETE ISLiM MODEL
# --------------------------------------------------

def load_islim():

    download_models()

    print("\nLoading ISLiM components...")

    encoder = load_torchscript(MODEL_DIR / "encoder_traced.ptl")
    t5_encoder = load_torchscript(MODEL_DIR / "t5_encoder.ptl")
    decoder_first = load_torchscript(MODEL_DIR / "t5_decoder_first_step.ptl")
    decoder_step = load_torchscript(MODEL_DIR / "t5_decoder_step.ptl")
    lm_head = load_torchscript(MODEL_DIR / "t5_lm_head.ptl")

    print("Loading T5 tokenizer...")

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    print("\nISLiM loaded successfully on CPU.")

    return {
        "encoder": encoder,
        "t5_encoder": t5_encoder,
        "decoder_first": decoder_first,
        "decoder_step": decoder_step,
        "lm_head": lm_head,
        "tokenizer": tokenizer,
        "device": DEVICE,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
