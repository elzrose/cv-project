
import numpy as np
from pathlib import Path

from src.islim.model_loader import load_islim
from src.islim.pipeline import translate_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    print("Loading ISLiM...")
    models = load_islim()

    print("\nLoading landmark sequence...")
    sequence = np.load(
        PROJECT_ROOT / "islm_sequence_interpolated.npy"
    )

    print("Sequence:", sequence.shape)

    print("\nRunning complete pipeline...")
    result = translate_sequence(sequence, models)

    print("\n========== RESULT ==========")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
