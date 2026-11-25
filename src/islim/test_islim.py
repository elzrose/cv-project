
import sys
from pathlib import Path
import numpy as np
import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.islim.model_loader import load_islim
from src.islim.preprocessing import prepare_islim_input
from src.islim.inference import translate_sign_keypoints


def main():
    print("=" * 50)
    print("ISLiM LOCAL INFERENCE TEST")
    print("=" * 50)

    # 1. Load model
    print("\n[1] Loading ISLiM...")
    models = load_islim()
    print("[OK] Models loaded.")

    # 2. Load an existing landmark sequence
    print("\n[2] Loading landmark sequence...")

    sequence_path = PROJECT_ROOT / "islm_sequence_interpolated.npy"

    if not sequence_path.exists():
        raise FileNotFoundError(
            f"Sequence not found: {sequence_path}"
        )

    sequence = np.load(sequence_path)
    print("Sequence shape:", sequence.shape)

    # 3. Preprocess
    print("\n[3] Preparing model input...")
    model_input, valid_frames = prepare_islim_input(sequence)

    print("Input shape:", tuple(model_input.shape))
    print("Valid frames:", valid_frames)

    # 4. Run translation
    print("\n[4] Translating...")
    prediction = translate_sign_keypoints(
        model_input,
        models
    )

    print("\n" + "=" * 50)
    print("TRANSLATION RESULT")
    print("=" * 50)
    print("Prediction:", prediction)
    print("=" * 50)


if __name__ == "__main__":
    main()
