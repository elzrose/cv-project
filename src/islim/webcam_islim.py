
# ============================================================
# HAND + POSE + FACE DETECTION + 94 LANDMARK RECORDING
# ============================================================

import cv2
import numpy as np
import time

from src.islim.model_loader import load_islim
from src.islim.pipeline import translate_sequence
from src.islim.mediapipe_setup import (
    mp, Handlandmarker, options,
    PoseLandmarker, pose_options,
    FaceLandmarker, face_options,
    extract_94_landmarks,
    finger_points, finger_pairs,
    calculate_distance, normalize_distance,
    calculate_angle, extract_features,
)
# ============================================================
# 1. LOAD ISLiM MODEL
# ============================================================

print("\nLoading ISLiM model...")
models = load_islim()
print("ISLiM ready! Start recording.")


# ============================================================
# 2. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open webcam.")

start_time = time.monotonic()
frame_count = 0
previous_time = time.time()


# ============================================================
# 2. INITIALIZE DETECTORS
# ============================================================

try:
    detector.close()
except (NameError, ValueError):
    pass

detector = Handlandmarker.create_from_options(options)


try:
    pose_detector.close()
except (NameError, ValueError):
    pass

pose_detector = PoseLandmarker.create_from_options(pose_options)


try:
    face_detector.close()
except (NameError, ValueError):
    pass

face_detector = FaceLandmarker.create_from_options(face_options)


# ============================================================
# 3. HAND CONNECTIONS
# ============================================================

connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# ============================================================
# 4. POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (28, 32)
]


# ============================================================
# 5. HAND FEATURE NAMES
# ============================================================

feature_names = [
    "Thumb_Angle",
    "Index_Angle",
    "Middle_Angle",
    "Ring_Angle",
    "Little_Angle",
    "Thumb_Index_Distance",
    "Index_Middle_Distance",
    "Middle_Ring_Distance",
    "Ring_Little_Distance"
]

latest_features = None
latest_blendshapes = None


# ============================================================
# 6. RECORDING VARIABLES
# ============================================================

# Stores one (94, 4) array for every recorded frame.
keypoint_sequence = []

# Recording state
recording = False


# ============================================================
# 7. SAVE FUNCTION
# ============================================================

def save_keypoint_sequence(sequence):

    if len(sequence) == 0:
        print("No frames were recorded.")
        return

    sequence_array = np.array(
        sequence,
        dtype=np.float32
    )

    print("\n========== SEQUENCE DETAILS ==========")

    print("Sequence shape:", sequence_array.shape)
    print("Data type:", sequence_array.dtype)
    print("Total frames:", sequence_array.shape[0])
    print("Landmarks per frame:", sequence_array.shape[1])
    print("Values per landmark:", sequence_array.shape[2])

    np.save("islm_sequence.npy", sequence_array)

    print("\nSaved successfully as islm_sequence.npy")


# ============================================================
# 8. MAIN WEBCAM LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        height, width, _ = frame.shape

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = frame_count * 33
        frame_count += 1


        # ====================================================
        # 9. HAND DETECTION
        # ====================================================

        result = detector.detect_for_video(
            mp_img,
            timestamp_ms
        )


        # ====================================================
        # 10. POSE DETECTION
        # ====================================================

        pose_result = pose_detector.detect_for_video(
            mp_img,
            timestamp_ms
        )

        if pose_result.pose_landmarks:

            pose_landmarks = pose_result.pose_landmarks[0]

            # Draw pose connections

            for start_idx, end_idx in POSE_CONNECTIONS:

                start_point = pose_landmarks[start_idx]
                end_point = pose_landmarks[end_idx]

                x1 = int(start_point.x * width)
                y1 = int(start_point.y * height)

                x2 = int(end_point.x * width)
                y2 = int(end_point.y * height)

                if (
                    0 <= x1 < width and
                    0 <= y1 < height and
                    0 <= x2 < width and
                    0 <= y2 < height
                ):

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

            # Draw pose landmarks

            for i, landmark in enumerate(pose_landmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                f"Pose: {len(pose_landmarks)}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


        # ====================================================
        # 11. FACE DETECTION
        # ====================================================

        face_result = face_detector.detect_for_video(
            mp_img,
            timestamp_ms
        )

        latest_blendshapes = None

        if face_result.face_landmarks:

            face_landmarks = face_result.face_landmarks[0]

            # Draw face landmarks

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if 0 <= x < width and 0 <= y < height:

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (255, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                f"Face: {len(face_landmarks)}",
                (10, height - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Read expression blendshapes

            if face_result.face_blendshapes:

                latest_blendshapes = face_result.face_blendshapes[0]

                cv2.putText(
                    frame,
                    f"Blendshapes: {len(latest_blendshapes)}",
                    (10, height - 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )


        # ====================================================
        # 12. EXTRACT 94 LANDMARKS
        # ====================================================

        frame_keypoints = extract_94_landmarks(
            result,
            pose_result,
            face_result
        )


        # ====================================================
        # 13. COLLECT LANDMARKS WHILE RECORDING
        # ====================================================

        if recording:

            keypoint_sequence.append(
                frame_keypoints.copy()
            )


        # ====================================================
        # 14. HAND FEATURE PROCESSING
        # ====================================================

        latest_features = None

        processing_time = time.time() - previous_time
        previous_time = time.time()

        for hand_index, handlandmarks in enumerate(result.hand_landmarks):

            # Draw hand landmarks

            for i, landmark in enumerate(handlandmarks):

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )

            wrist = [
                handlandmarks[0].x,
                handlandmarks[0].y
            ]

            middle_base = [
                handlandmarks[9].x,
                handlandmarks[9].y
            ]

            hand_size = calculate_distance(
                wrist,
                middle_base
            )

            # Draw hand connections

            for start, end in connections:

                x1 = int(handlandmarks[start].x * width)
                y1 = int(handlandmarks[start].y * height)

                x2 = int(handlandmarks[end].x * width)
                y2 = int(handlandmarks[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

            # Finger angles and status

            for finger_index, (finger, points) in enumerate(
                finger_points.items()
            ):

                a_idx, b_idx, c_idx = points

                a = [
                    handlandmarks[a_idx].x,
                    handlandmarks[a_idx].y
                ]

                b = [
                    handlandmarks[b_idx].x,
                    handlandmarks[b_idx].y
                ]

                c = [
                    handlandmarks[c_idx].x,
                    handlandmarks[c_idx].y
                ]

                angle = calculate_angle(a, b, c)

                status = "Extended" if angle > 160 else "Bent"

                x_position = 10 if hand_index == 0 else width - 220
                y_position = 30 + finger_index * 30

                cv2.putText(
                    frame,
                    f"{finger}: {status}",
                    (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            # Finger distances

            for pair_index, (pair_name, points) in enumerate(
                finger_pairs.items()
            ):

                point1, point2 = points

                a = [
                    handlandmarks[point1].x,
                    handlandmarks[point1].y
                ]

                b = [
                    handlandmarks[point2].x,
                    handlandmarks[point2].y
                ]

                distance = calculate_distance(a, b)

                normalized_distance = normalize_distance(
                    distance,
                    hand_size
                )

                x_position = 10 if hand_index == 0 else width - 220
                y_position = 210 + pair_index * 25

                cv2.putText(
                    frame,
                    f"{pair_name}: {normalized_distance:.3f}",
                    (x_position, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )

            # Extract nine hand features

            features = extract_features(handlandmarks)

            latest_features = features


        # ====================================================
        # 15. DISPLAY RECORDING STATUS
        # ====================================================

        if recording:

            cv2.putText(
                frame,
                f"RECORDING | Frames: {len(keypoint_sequence)}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "NOT RECORDING",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


        # ====================================================
        # 16. DISPLAY WEBCAM
        # ====================================================

        cv2.imshow("MediaPipe Detection", frame)

        key = cv2.waitKey(10) & 0xFF


        # ====================================================
        # 17. KEYBOARD CONTROLS
        # ====================================================

        # Press R to start or stop recording

        if key == ord('r'):

            if not recording:

                # START RECORDING

                keypoint_sequence = []
                recording = True

                print("\nRecording started...")

            else:

                # STOP RECORDING

                recording = False

                print("\nRecording stopped.")

                if len(keypoint_sequence) == 0:
                    print("No frames recorded.")

                else:
                    print("\nRunning ISLiM translation...")

                    sequence_array = np.asarray(
                        keypoint_sequence,
                        dtype=np.float32
                    )

                    try:
                        translation_result = translate_sequence(
                            sequence_array,
                            models
                        )

                        print("\n========== TRANSLATION RESULT ==========")
                        print("Prediction:", translation_result["prediction"])
                        print("Frames:", translation_result["input_frames"])
                        print("Inference time:", translation_result["inference_time"], "seconds")
                        print("Total time:", translation_result["total_time"], "seconds")
                        print("========================================")

                    except Exception as error:
                        print("\nISLiM translation failed:", error)

                    # Keep saving the recording for debugging.
                    save_keypoint_sequence(keypoint_sequence)


        # Press S to inspect the current frame

        elif key == ord('s'):

            print("\n========== CURRENT FRAME ==========")

            print("Shape:", frame_keypoints.shape)
            print("Data type:", frame_keypoints.dtype)

            print(
                "Pose:",
                np.sum(frame_keypoints[0:6, 3] > 0.5)
            )

            print(
                "Left hand:",
                np.sum(frame_keypoints[6:27, 3] > 0.5)
            )

            print(
                "Right hand:",
                np.sum(frame_keypoints[27:48, 3] > 0.5)
            )

            print(
                "Face:",
                np.sum(frame_keypoints[48:94, 3] > 0.5)
            )


        # Press Q to exit

        elif key == ord('q'):

            # Save unfinished recording before exiting

            if recording:

                recording = False

                print("\nSaving recording before exit...")

                save_keypoint_sequence(
                    keypoint_sequence
                )

            break


# ============================================================
# 18. CLEANUP
# ============================================================

finally:

    cap.release()

    cv2.destroyAllWindows()

    detector.close()
    pose_detector.close()
    face_detector.close()

    print("\nAll three detectors closed.")
