# ISL Translator 🇮🇳

**An experimental Indian Sign Language (ISL) to English video translation system using Computer Vision and Deep Learning.**

ISL Translator captures a signing video through a Chrome extension, extracts hand, body, and facial landmarks using MediaPipe, and passes the resulting landmark sequence to a pretrained ISLiM model to generate an English translation.

> **Project status: Experimental V1** — The complete video-to-translation pipeline is implemented and tested. Translation accuracy is still a work in progress.

## ✨ Features

- **Video recording:** Record signing videos directly through a Chrome extension.
- **Hand landmark detection:** Extract hand keypoints using MediaPipe.
- **Body landmark detection:** Capture relevant upper-body pose information.
- **Facial landmark detection:** Extract facial features that contribute to sign language communication.
- **Video preprocessing:** Convert extracted landmarks into the format expected by the pretrained model.
- **English translation:** Generate an English text prediction using the ISLiM model.
- **REST API:** Connect the Chrome extension to the Python backend through FastAPI.

## 🏗️ Architecture

```text
       Chrome Extension
              |
       Record Signing Video
              |
        Upload Video
              |
              v
        FastAPI Backend
              |
              v
       Video Processing
              |
              v
       MediaPipe Tasks
       /      |       \
    Hands    Pose     Face
       \      |       /
              |
              v
      94 Landmarks / Frame
              |
              v
       Preprocessing
              |
              v
        Pretrained ISLiM
              |
              v
       English Translation
              |
              v
       Display Prediction
       in Chrome Extension
```

## 🧠 Computer Vision Pipeline

The project uses MediaPipe Tasks to extract landmarks from each video frame.

The current landmark representation contains **94 landmarks per frame**, organized as:

| Component | Landmarks |
|---|---:|
| Upper-body pose | 6 |
| Both hands | 42 |
| Facial features | 46 |
| **Total** | **94** |

Each landmark is represented using four values: `x`, `y`, `z`, and visibility/confidence information.

The extracted sequence is preprocessed before being passed to the translation model.

## 🛠️ Tech Stack

| Area | Technologies |
|---|---|
| Computer Vision | MediaPipe, OpenCV |
| Machine Learning | PyTorch, ISLiM |
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Browser Integration | Chrome Extension (Manifest V3) |
| Data Processing | NumPy |

## 📁 Project Structure

```text
isl-cv-project/
│
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── style.css
│
├── src/
│   └── islim/
│       ├── api.py
│       ├── mediapipe_setup.py
│       ├── video_processing.py
│       ├── model_loader.py
│       ├── pipeline.py
│       └── ...
│
├── requirements.txt
├── .gitignore
└── README.md
```

Model checkpoints, MediaPipe task assets, generated data, and temporary video uploads are not included in the repository.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/elzrose/cv-project.git
cd cv-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

PyTorch installation may need to be configured separately depending on whether you want CPU or CUDA inference. For NVIDIA GPU support, install a compatible CUDA-enabled PyTorch build.

### 4. Download MediaPipe model assets

Download the following MediaPipe task models and place them in the repository root:

| Model | Download |
|---|---|
| Hand Landmarker | [Download](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) |
| Pose Landmarker Full | [Download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task) |
| Face Landmarker | [Download](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) |

The expected structure is:

```text
isl-cv-project/
├── face_landmarker.task
├── hand_landmarker.task
├── pose_landmarker_full.task
└── ...
```

### 5. Load the pretrained ISLiM model

The backend uses the pretrained ISLiM model. Its checkpoint files are downloaded by the project's model-loading code when required.

Ensure that you have an active internet connection for the initial model download.

### 6. Start the backend

From the repository root, run:

```bash
python -m uvicorn src.islim.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

You can check the backend at:

- Health check: `http://127.0.0.1:8000/health`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

## 🌐 Chrome Extension Setup

1. Open Google Chrome.
2. Navigate to `chrome://extensions`.
3. Enable **Developer mode**.
4. Click **Load unpacked**.
5. Select the repository's `extension` folder.
6. Pin the ISL Translator extension to your toolbar.

Make sure the FastAPI backend is running before using the extension.

### Using the extension

1. Open the ISL Translator extension.
2. Allow camera access when prompted.
3. Click **Start Recording** and perform the signs.
4. Click **Stop Recording**.
5. Preview the recorded video.
6. Click **Translate**.
7. View the generated English prediction.

## ⚠️ Current Limitations

This is an experimental implementation, and several limitations remain:

- **Translation accuracy:** The pretrained model does not consistently generate accurate translations. Some predictions may be incorrect or unrelated to the input video.
- **Full-video processing:** The current pipeline processes the complete recording rather than translating signs in real time.
- **Inference time:** Translation takes a few seconds on the tested local setup, depending on video length and hardware.
- **Model limitations:** The current implementation relies on a pretrained model; improving its compatibility with the extracted landmarks and translation quality is future work.

The system should not be used as a reliable communication or accessibility tool in its current state.

## 🔭 Future Improvements

- Improve translation accuracy through better model evaluation and experimentation.
- Explore fine-tuning and alternative ISL translation architectures.
- Investigate temporal and contextual information for more meaningful sentence generation.
- Optimize inference speed and memory usage.
- Explore real-time translation.
- Improve handling of facial expressions and non-manual sign components.

## 🙌 Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) for hand, pose, and face landmark detection.
- [ISLiM](https://github.com/ahmedshoaib/ISLiM) for the pretrained sign language translation model.

---

*Built as a computer vision learning project exploring Indian Sign Language recognition and video-to-text translation.*
