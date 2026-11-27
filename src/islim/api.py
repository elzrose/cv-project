
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool

from src.islim.model_loader import load_islim
from src.islim.video_processing import extract_video_landmarks
from src.islim.pipeline import translate_sequence


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm"}
MAX_FILE_SIZE = 100 * 1024 * 1024


# --------------------------------------------------
# 2. LOAD MODEL ONCE AT SERVER STARTUP
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nLoading ISLiM model...")

    app.state.models = await run_in_threadpool(load_islim)

    print("ISLiM is ready to receive videos!\n")

    yield

    app.state.models = None
    print("ISLiM model resources released.")


# --------------------------------------------------
# 3. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="ISL Translation API",
    description="Indian Sign Language Video-to-English Translation",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# 4. HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "running",
        "model_loaded": getattr(app.state, "models", None) is not None,
        "message": "ISL Translation Backend is working!"
    }


# --------------------------------------------------
# 5. COMPLETE VIDEO TRANSLATION
# --------------------------------------------------

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):

    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format"
        )

    saved_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    size = 0

    try:

        # A. Save uploaded video temporarily

        with saved_path.open("wb") as video:

            while True:

                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video exceeds 100 MB limit"
                    )

                video.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty"
            )

        # B. Extract MediaPipe landmarks

        print("\nExtracting video landmarks...")

        keypoints, video_metadata = await run_in_threadpool(
            extract_video_landmarks,
            saved_path
        )

        print("Landmark extraction completed!")
        print("Keypoint shape:", keypoints.shape)

        # C. Run preprocessing and ISLiM inference

        print("\nRunning ISLiM translation...")

        result = await run_in_threadpool(
            translate_sequence,
            keypoints,
            app.state.models
        )

        print("Translation completed:", result["prediction"])

        # D. Return translation and metadata

        return {
            "status": "success",
            "translation": result["prediction"],
            "video_metadata": video_metadata,
            "pipeline_metadata": result
        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    finally:

        await file.close()

        # E. Delete temporary video

        if saved_path.exists():
            saved_path.unlink()
