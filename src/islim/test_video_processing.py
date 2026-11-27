
from pathlib import Path

from src.islim.video_processing import extract_video_landmarks


# Use one of your recorded test videos
video_path = Path("uploads/0f2aebedfa7f4ea380346bf7a6b59416.mp4")

keypoints, metadata = extract_video_landmarks(video_path)

print("\n========== VIDEO PROCESSING RESULT ==========")

print("Metadata:", metadata)
print("Sequence shape:", keypoints.shape)
print("Data type:", keypoints.dtype)

print("First frame shape:", keypoints[0].shape)

print("=============================================")

# Save for inspection
import numpy as np

np.save("test_video_keypoints.npy", keypoints)

print("Landmarks saved successfully!")
