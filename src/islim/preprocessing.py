
import numpy as np
import torch

MAX_FRAMES = 128
NUM_LANDMARKS = 94
NUM_FEATURES = 4


def prepare_islim_input(keypoint_sequence):
    """
    Convert a landmark sequence from (T, 94, 4)
    into the ISLiM model input format (1, 4, 94, 128).
    """

    # 1. Work on a copy
    keypoints = np.asarray(
        keypoint_sequence,
        dtype=np.float32
    ).copy()

    # 2. Validate input
    if keypoints.ndim != 3:
        raise ValueError(
            f"Expected 3 dimensions (T, 94, 4), got {keypoints.shape}"
        )

    T, K, D = keypoints.shape

    if T == 0:
        raise ValueError("Keypoint sequence contains no frames.")

    if K != NUM_LANDMARKS:
        raise ValueError(f"Expected 94 landmarks, got {K}")

    if D != NUM_FEATURES:
        raise ValueError(f"Expected 4 features, got {D}")

    # 3. Normalize X, Y and Z
    for coord in range(3):
        arr = keypoints[:, :, coord]

        minimum = arr.min()
        maximum = arr.max()

        value_range = maximum - minimum

        if value_range > 1e-8:
            keypoints[:, :, coord] = (
                arr - minimum
            ) / value_range
        else:
            keypoints[:, :, coord] = arr - minimum

    # 4. Temporal sampling or padding
    original_frames = T

    if T > MAX_FRAMES:
        indices = torch.linspace(
            0,
            T - 1,
            MAX_FRAMES
        ).long().numpy()

        keypoints = keypoints[indices]

    elif T < MAX_FRAMES:
        padding = np.zeros(
            (MAX_FRAMES - T, NUM_LANDMARKS, NUM_FEATURES),
            dtype=np.float32
        )

        keypoints = np.concatenate(
            [keypoints, padding],
            axis=0
        )

    # 5. Rearrange dimensions:
    # (128, 94, 4) -> (4, 94, 128)
    keypoints = np.transpose(
        keypoints,
        (2, 1, 0)
    ).copy()

    # 6. Add batch dimension:
    # (4, 94, 128) -> (1, 4, 94, 128)
    keypoints = np.expand_dims(keypoints, axis=0)

    return (
        torch.from_numpy(keypoints).float(),
        min(original_frames, MAX_FRAMES)
    )
