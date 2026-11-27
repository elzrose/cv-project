
import cv2
import numpy as np
import mediapipe as mp

from src.islim.mediapipe_setup import (
    Handlandmarker,
    options,
    PoseLandmarker,
    pose_options,
    FaceLandmarker,
    face_options,
    extract_94_landmarks,
)


def extract_video_landmarks(video_path):
    """
    Process a recorded video using MediaPipe.

    Returns:
        keypoint_sequence: NumPy array of shape (T, 94, 4)
        metadata: Information about the processed video
    """

    # 1. Open the uploaded video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keypoint_sequence = []

    # 2. Create MediaPipe detectors
    hand_detector = Handlandmarker.create_from_options(options)
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    face_detector = FaceLandmarker.create_from_options(face_options)

    frame_index = 0
    previous_timestamp = -1

    try:
        while cap.isOpened():

            # 3. Read the next frame
            success, frame = cap.read()

            if not success:
                break

            # 4. Convert OpenCV BGR image to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # 5. Create MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # 6. Generate monotonically increasing timestamps
            timestamp_ms = max(
                int(frame_index * 1000 / fps),
                previous_timestamp + 1
            )
            previous_timestamp = timestamp_ms

            # 7. Run all three detectors
            hand_result = hand_detector.detect_for_video(
                mp_image, timestamp_ms
            )

            pose_result = pose_detector.detect_for_video(
                mp_image, timestamp_ms
            )

            face_result = face_detector.detect_for_video(
                mp_image, timestamp_ms
            )

            # 8. Extract the 94 landmarks
            keypoints = extract_94_landmarks(
                hand_result,
                pose_result,
                face_result
            )

            keypoint_sequence.append(keypoints)

            frame_index += 1

    finally:
        # 9. Release all resources
        cap.release()
        hand_detector.close()
        pose_detector.close()
        face_detector.close()

    if not keypoint_sequence:
        raise ValueError("No frames could be extracted from the video.")

    # 10. Convert list of frames into NumPy array
    keypoint_sequence = np.stack(
        keypoint_sequence,
        axis=0
    ).astype(np.float32)

    metadata = {
        "fps": round(float(fps), 2),
        "total_frames": total_frames,
        "processed_frames": len(keypoint_sequence),
        "duration_seconds": round(len(keypoint_sequence) / fps, 2),
        "landmark_shape": keypoint_sequence.shape,
    }

    return keypoint_sequence, metadata
