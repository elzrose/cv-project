
const API_URL = "http://127.0.0.1:8000";
const cameraPreview = document.getElementById("cameraPreview");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraStatus = document.getElementById("cameraStatus");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const translateBtn = document.getElementById("translateBtn");

const recordingIndicator = document.getElementById("recordingIndicator");
const statusMessage = document.getElementById("statusMessage");
const translationResult = document.getElementById("translationResult");

// Recording state
let cameraStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recordedUrl = null;


// --------------------------------------------------
// 1. START CAMERA AND RECORDING
// --------------------------------------------------

startBtn.addEventListener("click", async () => {

    try {
        // Reset previous recording
        recordedBlob = null;
        recordedChunks = [];

        translateBtn.disabled = true;
        translationResult.textContent = "Your translation will appear here...";
        statusMessage.textContent = "Requesting camera access...";

        if (recordedUrl) {
            URL.revokeObjectURL(recordedUrl);
            recordedUrl = null;
        }

        cameraPreview.pause();
        cameraPreview.removeAttribute("src");
        cameraPreview.load();
        cameraPreview.controls = false;

        // Request webcam access
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });

        // Display webcam preview
        cameraPreview.srcObject = cameraStream;
        cameraPreview.style.display = "block";
        cameraPlaceholder.style.display = "none";

        cameraStatus.textContent = "Camera connected";
        statusMessage.textContent = "Recording in progress...";

        // Choose a supported recording format
        const mimeTypes = [
            "video/webm;codecs=vp9",
            "video/webm;codecs=vp8",
            "video/webm"
        ];

        const selectedMimeType = mimeTypes.find(type =>
            MediaRecorder.isTypeSupported(type)
        );

        const options = selectedMimeType
            ? { mimeType: selectedMimeType }
            : {};

        // Create recorder
        mediaRecorder = new MediaRecorder(cameraStream, options);

        // Collect recorded video chunks
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        mediaRecorder.onerror = (event) => {
        console.error("Recording error:", event.error);

        if (cameraStream) {
         cameraStream.getTracks().forEach(track => track.stop());
         cameraStream = null;
        }

        recordingIndicator.style.display = "none";
        startBtn.disabled = false;
        stopBtn.disabled = true;
        translateBtn.disabled = true;

        statusMessage.textContent = "Recording failed. Please try again.";
    };
        // Handle recording completion
        mediaRecorder.onstop = () => {

            recordedBlob = new Blob(recordedChunks, {
                type: mediaRecorder.mimeType || "video/webm"
            });

            // Release webcam
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
            cameraPreview.srcObject = null;

            // Show recorded video preview
            recordedUrl = URL.createObjectURL(recordedBlob);

            cameraPreview.src = recordedUrl;
            cameraPreview.controls = true;
            cameraPreview.style.display = "block";

            cameraStatus.textContent = "Recording completed";

            recordingIndicator.style.display = "none";

            startBtn.disabled = false;
            stopBtn.disabled = true;
            translateBtn.disabled = recordedBlob.size === 0;

            statusMessage.textContent =
                `Recording ready (${(recordedBlob.size / 1024 / 1024).toFixed(2)} MB)`;

            if (recordedBlob.size === 0) {
                statusMessage.textContent = "Recording failed. Please try again.";
            }
        };

        // Start recording
        mediaRecorder.start(250);

        startBtn.disabled = true;
        stopBtn.disabled = false;

        recordingIndicator.style.display = "block";

    } catch (error) {

        console.error("Camera error:", error);

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        cameraPreview.srcObject = null;
        cameraPreview.style.display = "none";
        cameraPlaceholder.style.display = "flex";

        cameraStatus.textContent = "Camera not connected";
        statusMessage.textContent =
            error.name === "NotAllowedError"
                ? "Camera permission denied. Allow camera access in Chrome."
                : "Could not access camera. Check your webcam.";

        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
});


// --------------------------------------------------
// 2. STOP RECORDING
// --------------------------------------------------

stopBtn.addEventListener("click", () => {

    if (mediaRecorder && mediaRecorder.state === "recording") {

        mediaRecorder.stop();

        stopBtn.disabled = true;
        statusMessage.textContent = "Preparing recorded video...";
    }
});


// --------------------------------------------------
// 3. UPLOAD VIDEO AND TRANSLATE
// --------------------------------------------------

translateBtn.addEventListener("click", async () => {

    if (!recordedBlob) {
        statusMessage.textContent = "Record a video first.";
        return;
    }

    translateBtn.disabled = true;
    startBtn.disabled = true;

    statusMessage.textContent = "Uploading video and translating...";
    translationResult.textContent = "Processing...";

    try {

        const formData = new FormData();

        formData.append(
            "file",
            recordedBlob,
            "isl-recording.webm"
        );

        const response = await fetch(
            `${API_URL}/translate`,
            {
                method: "POST",
                body: formData
            }
        );

        let data;

        try {
          data = await response.json();
        } catch {
          throw new Error("The backend returned an invalid response.");
        }

        if (!response.ok) {
            throw new Error(data.detail || "Translation failed.");
        }

        // Display English translation
        translationResult.textContent = data.translation;

        statusMessage.textContent = "Translation completed!";

        console.log("Translation response:", data);

    } catch (error) {

        console.error("Translation error:", error);

        translationResult.textContent = "Could not translate the video.";

        statusMessage.textContent = error.message;

    } finally {

        translateBtn.disabled = false;
        startBtn.disabled = false;
    }
});
