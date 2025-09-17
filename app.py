import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import traceback
import gc
import logging
from threading import Lock, Thread
import time
import base64

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

# ---------- Model and inference settings ----------
model = None
model_lock = Lock()

# Default image size — we'll try to override from model input if available
IMG_SIZE = (128, 128)
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 5))   # default: 5 frames per video
EARLY_STOP_CONFIDENCE = float(os.environ.get("EARLY_STOP_CONF", 0.85))
VIOLENCE_THRESHOLD = float(os.environ.get("VIOLENCE_THRESHOLD", 0.5))
FILE_CLEANUP_DELAY = int(os.environ.get("FILE_CLEANUP_DELAY", 60 * 60))  # 1 hour


# ---------- Model loading ----------
def load_model_safely():
    """
    Load model if present. If loaded, attempt to infer required IMG_SIZE
    from the model's input_shape and override global IMG_SIZE.
    """
    global model, IMG_SIZE
    try:
        model_path = "violence_detection_cnn.h5"
        if os.path.exists(model_path):
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            logger.info("Model loaded successfully from %s", model_path)

            # Try to infer input size from model.input_shape
            try:
                input_shape = getattr(model, "input_shape", None)
                if input_shape:
                    # input_shape may be (None, H, W, C) or ((None, H, W, C), ...)
                    if isinstance(input_shape, tuple) and len(input_shape) >= 3:
                        # handle nested shapes
                        if isinstance(input_shape[0], tuple):
                            s = input_shape[0]
                        else:
                            s = input_shape
                        if len(s) >= 4 and s[1] and s[2]:
                            IMG_SIZE = (int(s[1]), int(s[2]))
                            logger.info("Detected model input size: %s", IMG_SIZE)
                        else:
                            logger.info("Model input_shape present but not standard: %s", s)
                    else:
                        logger.info("Model input_shape not usable: %s", input_shape)
            except Exception as e:
                logger.warning("Could not parse model input_shape: %s", e)

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
            # returns shape (N,1) random
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
    - resize frames to IMG_SIZE (inferred from model)
    - sample up to MAX_SAMPLES frames evenly across duration
    - early stop if very high-confidence violence appears
    Returns a dict (prediction, confidence, max_confidence, violence_timestamps, total_samples, video_duration, error)
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

        # Use MAX_SAMPLES spread evenly across the whole duration
        num_samples = min(MAX_SAMPLES, max(1, int(duration)))
        # If video shorter than num_samples seconds, fallback to 1 sample per second up to duration
        if num_samples <= 0:
            num_samples = 1

        sample_points = np.linspace(0, duration, num=num_samples, endpoint=False)

        predictions = []
        violence_detected_times = []

        for ts in sample_points:
            timestamp_sec = int(ts)
            cap.set(cv2.CAP_PROP_POS_MSEC, int(timestamp_sec * 1000))
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.debug("Frame read failed at %ds", timestamp_sec)
                continue

            try:
                # Ensure frame is BGR; convert to RGB only if model used RGB training
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized.astype(np.float32) / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)

                with model_lock:
                    pred_arr = model.predict(frame_input, verbose=0)
                pred = float(pred_arr.ravel()[0])

                predictions.append(pred)

                if pred > VIOLENCE_THRESHOLD:
                    violence_detected_times.append({
                        "timestamp": timestamp_sec,
                        "confidence": float(pred),
                        "time_formatted": f"{int(timestamp_sec//60):02d}:{int(timestamp_sec%60):02d}"
                    })

                if pred >= EARLY_STOP_CONFIDENCE:
                    logger.info("Early stop at %ds with pred=%.3f", timestamp_sec, pred)
                    break

                # GC occasionally
                if len(predictions) % 3 == 0:
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
            file = request.files.get('video')
            if not file or file.filename == '':
                return render_template("result.html", prediction="Error: No file uploaded", video_path=None, details=None)

            if not allowed_file(file.filename):
                return render_template("result.html", prediction="Error: Unsupported video type", video_path=None, details=None)

            filename_unique = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename_unique)
            try:
                file.save(filepath)
                logger.info("Saved upload to %s", filepath)
            except Exception as e:
                logger.exception("Failed to save file: %s", e)
                return render_template("result.html", prediction=f"Error saving file: {e}", video_path=None, details=None)

            logger.info("Starting video analysis for %s", filename_unique)
            start_ts = time.time()
            result = predict_video_optimized(filepath)
            elapsed = time.time() - start_ts
            logger.info("Analysis finished in %.2fs: %s", elapsed, str(result.get("prediction", result.get("error"))))

            # schedule cleanup (so playback works on result page now)
            schedule_file_cleanup(filepath, delay=FILE_CLEANUP_DELAY)

            if result.get("error"):
                return render_template("result.html", prediction=result["error"], video_path=None, details=None)

            return render_template("result.html", prediction=result["prediction"], video_path=filename_unique, details=result)

        except Exception as e:
            logger.exception("Upload route error: %s", e)
            traceback.print_exc()
            return render_template("result.html", prediction=f"Server error: {e}", video_path=None, details=None)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """
    Accepts JSON: { image: "data:image/jpeg;base64,..." }
    Returns JSON: { violence_detected: bool, confidence: float, threat_level: "High"|"Low" }
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400

        image_b64 = data['image']
        if ',' in image_b64:
            image_data = image_b64.split(',', 1)[1]
        else:
            image_data = image_b64

        try:
            image_bytes = base64.b64decode(image_data)
        except Exception:
            return jsonify({"error": "Invalid base64 image data"}), 400

        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        try:
            frame_resized = cv2.resize(frame, IMG_SIZE)
        except Exception as e:
            logger.exception("Resize failed: %s", e)
            return jsonify({"error": f"Resize failed: {e}"}), 500

        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_input = np.expand_dims(frame_norm, axis=0)

        with model_lock:
            pred_arr = model.predict(frame_input, verbose=0)
        pred = float(pred_arr.ravel()[0])

        violence_detected = pred > VIOLENCE_THRESHOLD
        confidence_pct = round(float(pred) * 100, 2)

        return jsonify({
            "violence_detected": bool(violence_detected),
            "confidence": confidence_pct,
            "threat_level": "High" if violence_detected else "Low"
        })

    except Exception as e:
        logger.exception("Error in /detect_frame: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_input_size": IMG_SIZE,
        "timestamp": time.time()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
