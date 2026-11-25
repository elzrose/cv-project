
import time
import numpy as np

from src.islim.preprocessing import prepare_islim_input
from src.islim.inference import translate_sign_keypoints


def translate_sequence(keypoint_sequence, models):
    """
    Complete pipeline:
    MediaPipe landmarks -> preprocessing -> ISLiM -> English text.

    Input:
        keypoint_sequence: (T, 94, 4)

    Output:
        Dictionary containing prediction and timing information.
    """

    start_time = time.time()

    # 1. Validate input
    keypoint_sequence = np.asarray(
        keypoint_sequence,
        dtype=np.float32
    )

    if keypoint_sequence.ndim != 3:
        raise ValueError(
            f"Expected (T, 94, 4), got {keypoint_sequence.shape}"
        )

    # 2. Preprocess landmarks
    model_input, valid_frames = prepare_islim_input(
        keypoint_sequence
    )

    # 3. Run ISLiM inference
    inference_start = time.time()

    prediction = translate_sign_keypoints(
        model_input,
        models
    )

    inference_time = time.time() - inference_start
    total_time = time.time() - start_time

    return {
        "prediction": prediction,
        "input_frames": len(keypoint_sequence),
        "valid_frames": valid_frames,
        "input_shape": tuple(model_input.shape),
        "inference_time": round(inference_time, 3),
        "total_time": round(total_time, 3)
    }
