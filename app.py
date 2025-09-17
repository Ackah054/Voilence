import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
import traceback
import gc
import logging
from threading import Lock, Thread
import time

# ---------- Config ----------
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 50))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024  # MB -> bytes

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model and inference settings (tune via env if needed)
model = None
model_lock = Lock()
IMG_SIZE = (64, 64)                        # smaller image reduces inference cost
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 6))     # total frames sampled per video
MAX_WORKER_TIMEOUT_S = int(os.environ.get("WORKER_TIMEOUT", 120))
EARLY_STOP_CONFIDENCE = float(os.environ.get("EARLY_STOP_CONF", 0.85))
VIOLENCE_THRESHOLD = float(os.environ.get("VIOLENCE_THRESHOLD", 0.5))
FILE_CLEANUP_DELAY = int(os.environ.get("FILE_CLEANUP_DELAY", 60 * 60))  # seconds (default 1 hour)


# ---------- Model loading ----------
def load_model_safely():
    """Load real model if available. Return True if loaded, False otherwise."""
    global model
    try:
        model_path = "violence_detection_cnn.h5"
        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            logger.info("Model loaded successfully from %s", model_path)
            return True
        else:
            logger.warning("Model file not found at %s", model_path)
            return False
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return False


def create_dummy_model():
    """A tiny dummy model for testing (returns random scores)."""
    global model

    class DummyModel:
        def predict(self, x, verbose=0):
            # returns shape (N,1)
            return np.random.random((x.shape[0], 1))

    model = DummyModel()
    logger.warning("Using dummy model (random predictions). Replace with real model for production.")


if not load_model_safely():
    create_dummy_model()


# ---------- Helpers ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def schedule_file_cleanup(path, delay=FILE_CLEANUP_DELAY):
    """Delete the uploaded file after `delay` seconds in a daemon thread."""
    def _cleanup():
        try:
            time.sleep(delay)
            if os.path.exists(path):
                os.remove(path)
                logger.info("Scheduled cleanup removed file: %s", path)
        except Exception as e:
            logger.warning("Scheduled cleanup failed for %s: %s", path, e)

    t = Thread(target=_cleanup, daemon=True)
    t.start()


# ---------- Video inference ----------
def predict_video_optimized(video_path):
    """
    Predict violence in a video while keeping computation light:
     - resize frames to IMG_SIZE
     - sample up to MAX_SAMPLES frames spread across duration
     - early stop if a very high-confidence violence frame appears
    Returns dict with keys: prediction, confidence, max_confidence, violence_timestamps, total_samples, video_duration, error
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file", "violence_timestamps": []}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frame_count / fps if fps > 0 else 0.0
        logger.info("Video properties: duration=%.2fs, frames=%d, fps=%d", duration, frame_count, fps)

        if duration <= 0 or frame_count <= 0:
            return {"error": "Video contains no readable frames", "violence_timestamps": []}

        # determine samples spread evenly across the whole video
        num_samples = min(MAX_SAMPLES, max(1, int(duration)))  # ensure at least 1
        # if duration shorter than MAX_SAMPLES, sample one per second up to duration
        # use linspace to spread samples
        sample_points = np.linspace(0, duration, num=num_samples, endpoint=False)
        predictions = []
        timestamps = []
        violence_detected_times = []

        for i, ts in enumerate(sample_points):
            timestamp_sec = int(ts)
            # Jump to timestamp (ms)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Frame read failed at %ds (sample %d)", timestamp_sec, i)
                continue

            try:
                # resize & normalize (keep 3 channels - model likely trained on RGB)
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized.astype(np.float32) / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)  # shape (1,H,W,3)

                # Predict under lock
                with model_lock:
                    pred_arr = model.predict(frame_input, verbose=0)
                # support different output shapes
                pred = float(pred_arr.ravel()[0])

                predictions.append(pred)
                timestamps.append(timestamp_sec)

                if pred > VIOLENCE_THRESHOLD:
                    violence_detected_times.append({
                        "timestamp": timestamp_sec,
                        "confidence": float(pred),
                        "time_formatted": f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                    })

                # Early-stop on very strong signal
                if pred >= EARLY_STOP_CONFIDENCE:
                    logger.info("Early stop at %ds with pred=%.3f", timestamp_sec, pred)
                    break

                # periodic GC to keep memory low
                if i % 3 == 0:
                    gc.collect()

            except MemoryError as me:
                logger.exception("MemoryError while processing frame at %ds: %s", timestamp_sec, me)
                return {"error": "Server out of memory while processing video", "violence_timestamps": []}
            except Exception as e:
                logger.exception("Frame processing error at %ds: %s", timestamp_sec, e)
                continue

        if len(predictions) == 0:
            return {"error": "No frames could be processed", "violence_timestamps": []}

        avg_pred = float(np.mean(predictions))
        max_pred = float(np.max(predictions))

        return {
            "prediction": "Violence Detected" if avg_pred > VIOLENCE_THRESHOLD else "No Violence Detected",
            "confidence": avg_pred,
            "max_confidence": max_pred,
            "violence_timestamps": violence_detected_times,
            "total_samples": len(predictions),
            "video_duration": duration,
            "error": None
        }

    except Exception as e:
        logger.exception("Video processing failed: %s", e)
        return {"error": f"Processing failed: {e}", "violence_timestamps": []}
    finally:
        if cap:
            cap.release()
        gc.collect()


# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                return render_template("result.html",
                                       prediction="Error: No file uploaded",
                                       video_path=None,
                                       details=None)

            file = request.files['video']
            if file.filename == '':
                return render_template("result.html",
                                       prediction="Error: No file selected",
                                       video_path=None,
                                       details=None)

            if not allowed_file(file.filename):
                return render_template("result.html",
                                       prediction="Error: Unsupported video type",
                                       video_path=None,
                                       details=None)

            filename = secure_filename(file.filename)
            # make filename unique to avoid collisions (timestamp)
            filename_unique = f"{int(time.time())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_unique)

            try:
                file.save(filepath)
                logger.info("Saved upload to %s", filepath)
            except Exception as e:
                logger.exception("Failed to save file: %s", e)
                return render_template("result.html",
                                       prediction=f"Error saving file: {e}",
                                       video_path=None,
                                       details=None)

            # analyze (synchronous, but light)
            logger.info("Starting video analysis for %s", filename_unique)
            start_ts = time.time()
            result = predict_video_optimized(filepath)
            elapsed = time.time() - start_ts
            logger.info("Analysis finished in %.2fs: %s", elapsed, str(result.get("prediction", result.get("error"))))

            if result.get("error"):
                # schedule cleanup anyway
                schedule_file_cleanup(filepath)
                return render_template("result.html",
                                       prediction=result["error"],
                                       video_path=None,
                                       details=None)

            # schedule file to be deleted after delay so result page can play it
            schedule_file_cleanup(filepath, delay=FILE_CLEANUP_DELAY)

            # pass details to result page (result contains prediction + timestamps)
            return render_template("result.html",
                                   prediction=result["prediction"],
                                   video_path=filename_unique,
                                   details=result)

        except Exception as e:
            logger.exception("Upload route error: %s", e)
            traceback.print_exc()
            return render_template("result.html",
                                   prediction=f"Server error: {e}",
                                   video_path=None,
                                   details=None)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/camera_feed')
def camera_feed():
    return jsonify({"error": "Camera feed not available on cloud deployment"})


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # run with debug=False in production; gunicorn will call app:app
    app.run(host='0.0.0.0', port=port, debug=False)
