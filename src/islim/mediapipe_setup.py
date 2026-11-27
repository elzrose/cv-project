"""MediaPipe Tasks configuration and the existing 94-landmark extraction logic.

Run the application from the repository root with:
    python -m src.islim.webcam_islim
"""

from pathlib import Path

import numpy as np
import mediapipe as mp

# Resolve model assets from the repository root, not the current working folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# MediaPipe Tasks aliases and detector options (preserved from the notebook)
# ---------------------------------------------------------------------------
Baseoptions = mp.tasks.BaseOptions
Handlandmarker = mp.tasks.vision.HandLandmarker
HandlandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandlandmarkerOptions(
    base_options=Baseoptions(
        model_asset_path=str(PROJECT_ROOT / "hand_landmarker.task")
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(PROJECT_ROOT / "pose_landmarker_full.task")
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(PROJECT_ROOT / "face_landmarker.task")
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
)

# ---------------------------------------------------------------------------
# Landmark indices: 6 pose + 42 hands + 46 face = 94 landmarks
# ---------------------------------------------------------------------------
POSE_INDICES = [11, 12, 13, 14, 23, 24]

FACE_LIPS = [
    61, 146, 91, 181, 84,
    17, 314, 405, 321, 375,
    291, 409, 270, 95, 88,
    178, 87, 14, 78, 308,
]

FACE_EYES = [
    33, 133, 160, 158,
    153, 144, 145, 246,
    362, 263, 387, 385,
    380, 373, 374, 249,
]

FACE_EYEBROWS = [
    107, 66, 105, 63, 70,
    336, 296, 334, 293, 300,
]

FACE_INDICES = FACE_LIPS + FACE_EYES + FACE_EYEBROWS


def extract_94_landmarks(hand_result, pose_result, face_result):
    """Extract the notebook's ordered (94, 4) x/y/z/visibility array."""
    keypoints = np.zeros((94, 4), dtype=np.float32)

    # 1. Pose landmarks (0â€“5)
    if pose_result.pose_landmarks:
        pose_landmarks = pose_result.pose_landmarks[0]
        for i, idx in enumerate(POSE_INDICES):
            lm = pose_landmarks[idx]
            visibility = lm.visibility if lm.visibility is not None else 0.0
            keypoints[i] = [lm.x, lm.y, lm.z, visibility]

    # 2. Hand landmarks (6â€“47)
    if hand_result.hand_landmarks:
        for hand_index, hand_landmarks in enumerate(hand_result.hand_landmarks):
            if hand_index >= len(hand_result.handedness):
                continue

            classification = hand_result.handedness[hand_index][0]
            label = classification.category_name

            # Preserve the notebook's handedness swap for unmirrored webcam input.
            if label == "Left":
                label = "Right"
            elif label == "Right":
                label = "Left"

            if label == "Left":
                start_idx = 6
            elif label == "Right":
                start_idx = 27
            else:
                continue

            for i, lm in enumerate(hand_landmarks):
                keypoints[start_idx + i] = [lm.x, lm.y, lm.z, 1.0]

    # 3. Face landmarks (48â€“93)
    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        for i, idx in enumerate(FACE_INDICES):
            lm = face_landmarks[idx]
            visibility = 1.0 if lm.z > -0.5 else 0.5
            keypoints[48 + i] = [lm.x, lm.y, lm.z, visibility]

    return keypoints

# ---------------------------------------------------------------------------
# Existing hand-feature helper functions used by the webcam display
# ---------------------------------------------------------------------------
finger_points = {
    "Thumb": (1, 2, 4),
    "Index": (5, 6, 8),
    "Middle": (9, 10, 12),
    "Ring": (13, 14, 16),
    "Little": (17, 18, 20),
}

finger_pairs = {
    "Thumb-Index": (4, 8),
    "Index-Middle": (8, 12),
    "Middle-Ring": (12, 16),
    "Ring-Little": (16, 20),
}


def calculate_distance(point1, point2):
    return np.linalg.norm(
        np.array(point1, dtype=np.float32)
        - np.array(point2, dtype=np.float32)
    )


def normalize_distance(distance, hand_size):
    if hand_size < 1e-8:
        return 0.0
    return distance / hand_size


def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def extract_features(hand_landmarks):
    wrist = [hand_landmarks[0].x, hand_landmarks[0].y]
    middle_base = [hand_landmarks[9].x, hand_landmarks[9].y]
    hand_size = calculate_distance(wrist, middle_base)
    features = []

    for points in finger_points.values():
        a_idx, b_idx, c_idx = points
        a = [hand_landmarks[a_idx].x, hand_landmarks[a_idx].y]
        b = [hand_landmarks[b_idx].x, hand_landmarks[b_idx].y]
        c = [hand_landmarks[c_idx].x, hand_landmarks[c_idx].y]
        features.append(calculate_angle(a, b, c))

    for points in finger_pairs.values():
        p1, p2 = points
        a = [hand_landmarks[p1].x, hand_landmarks[p1].y]
        b = [hand_landmarks[p2].x, hand_landmarks[p2].y]
        distance = calculate_distance(a, b)
        features.append(normalize_distance(distance, hand_size))

    return features
