
"""
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
import traceback
import gc
import logging
from threading import Lock
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Reduced to 50 MB for better performance

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model loading
model = None
model_lock = Lock()
IMG_SIZE = (128, 128)

def load_model_safely():
    ""Load the model with proper error handling.""
    global model
    try:
        if os.path.exists("violence_detection_cnn.h5"):
            from tensorflow.keras.models import load_model
            model = load_model("violence_detection_cnn.h5")
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error("Model file 'violence_detection_cnn.h5' not found")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def create_dummy_model():
    ""Create a dummy model for testing when real model is not available.""
    global model
    
    class DummyModel:
        def predict(self, x, verbose=0):
            # Return random prediction for testing
            return np.array([[np.random.random()]])
    
    model = DummyModel()
    logger.warning("Using dummy model for testing - replace with real model for production")

if not load_model_safely():
    create_dummy_model()

def allowed_file(filename):
    ""Check if file extension is allowed.""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_video_optimized(video_path):
    ""Optimized video prediction with memory management and timeout handling.""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file", "timestamps": []}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Processing video: {duration:.2f}s, {frame_count} frames, {fps} fps")

        if duration == 0 or frame_count == 0:
            return {"error": "Video contains no readable frames", "timestamps": []}

        predictions = []
        timestamps = []
        violence_detected_times = []
        
        sample_interval = 2  # seconds
        max_samples = min(30, int(duration / sample_interval))  # Limit to 30 samples max
        
        for i in range(max_samples):
            timestamp = i * sample_interval
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                # Process frame
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized.astype(np.float32) / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)
                
                with model_lock:
                    pred = model.predict(frame_input, verbose=0)[0][0]
                
                predictions.append(float(pred))
                timestamps.append(timestamp)
                
                if pred > 0.5:
                    violence_detected_times.append({
                        "timestamp": timestamp,
                        "confidence": float(pred),
                        "time_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                    })
                
                # Force garbage collection every 10 frames
                if i % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Frame processing error at {timestamp}s: {e}")
                continue

        if len(predictions) == 0:
            return {"error": "No frames could be processed", "timestamps": []}

        avg_pred = float(np.mean(predictions))
        max_pred = float(np.max(predictions))
        
        result = {
            "prediction": "Violence Detected" if avg_pred > 0.5 else "No Violence Detected",
            "confidence": avg_pred,
            "max_confidence": max_pred,
            "violence_timestamps": violence_detected_times,
            "total_samples": len(predictions),
            "video_duration": duration,
            "error": None
        }
        
        return result

    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        return {"error": f"Processing failed: {str(e)}", "timestamps": []}
    
    finally:
        if cap:
            cap.release()
        gc.collect()

# Routes
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
                                     prediction="Error: Only video files are allowed (.mp4, .avi, .mov, .wmv)", 
                                     video_path=None, 
                                     details=None)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                logger.info(f"File saved: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save file: {e}")
                return render_template("result.html", 
                                     prediction=f"Error saving file: {str(e)}", 
                                     video_path=None, 
                                     details=None)

            logger.info("Starting video analysis...")
            start_time = time.time()
            
            result = predict_video_optimized(filepath)
            
            processing_time = time.time() - start_time
            logger.info(f"Video analysis completed in {processing_time:.2f}s")
            
            # Clean up uploaded file to save space
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to clean up file: {e}")

            if result.get("error"):
                return render_template('result.html', 
                                     prediction=result["error"], 
                                     video_path=None, 
                                     details=None)
            
            return render_template('result.html', 
                                 prediction=result["prediction"], 
                                 video_path=filename,
                                 details=result)

        except Exception as e:
            logger.error(f"Upload route error: {str(e)}")
            traceback.print_exc()
            return render_template('result.html', 
                                 prediction=f"Server error: {str(e)}", 
                                 video_path=None, 
                                 details=None)

    return render_template('upload.html')

@app.route('/camera')
def camera():
    ""Camera page template.""
    return render_template('camera.html')

@app.route('/camera_feed')
def camera_feed():
    ""Disabled for deployment - webcam not available on cloud servers.""
    return jsonify({"error": "Camera feed not available on cloud deployment"})

@app.route('/health')
def health_check():
    ""Health check endpoint for deployment monitoring.""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
"""





import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
import traceback
import gc
import logging
from threading import Lock
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
model = None
model_lock = Lock()
IMG_SIZE = (128, 128)


def load_model_safely():
    """Try to load the real model, else fallback to dummy."""
    global model
    try:
        if os.path.exists("violence_detection_cnn.h5"):
            from tensorflow.keras.models import load_model
            model = load_model("violence_detection_cnn.h5")
            logger.info("Model loaded successfully")
            return True
        else:
            logger.error("Model file not found")
            return False
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return False


def create_dummy_model():
    """Create a dummy model that returns random predictions (for testing)."""
    global model

    class DummyModel:
        def predict(self, x, verbose=0):
            return np.array([[np.random.random()]])

    model = DummyModel()
    logger.warning("Using dummy model (testing only)")


if not load_model_safely():
    create_dummy_model()


def allowed_file(filename):
    """Check if the file is an allowed video format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_video_optimized(video_path):
    """Predict violence in video with fewer samples and early stopping."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file", "timestamps": []}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        duration = frame_count / fps if fps > 0 else 0
        logger.info(f"Processing video: {duration:.2f}s, {frame_count} frames, {fps} fps")

        if duration == 0 or frame_count == 0:
            return {"error": "Video contains no readable frames", "timestamps": []}

        predictions, timestamps, violence_detected_times = [], [], []

        # Tune sampling to avoid timeouts
        sample_interval = 5  # every 5 seconds
        max_samples = min(12, int(duration / sample_interval))  # at most 12 frames

        for i in range(max_samples):
            timestamp = i * sample_interval
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            try:
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized.astype(np.float32) / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)

                with model_lock:
                    pred = model.predict(frame_input, verbose=0)[0][0]

                predictions.append(float(pred))
                timestamps.append(timestamp)

                if pred > 0.5:
                    violence_detected_times.append({
                        "timestamp": timestamp,
                        "confidence": float(pred),
                        "time_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
                    })

                    # EARLY STOP: If confidence is very high, stop to save time
                    if pred > 0.85:
                        logger.info("High confidence violence detected, stopping early")
                        break

                if i % 5 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Frame processing error at {timestamp}s: {e}")
                continue

        if not predictions:
            return {"error": "No frames could be processed", "timestamps": []}

        avg_pred = float(np.mean(predictions))
        max_pred = float(np.max(predictions))

        return {
            "prediction": "Violence Detected" if avg_pred > 0.5 else "No Violence Detected",
            "confidence": avg_pred,
            "max_confidence": max_pred,
            "violence_timestamps": violence_detected_times,
            "total_samples": len(predictions),
            "video_duration": duration,
            "error": None
        }

    except Exception as e:
        logger.error(f"Video processing error: {e}")
        return {"error": f"Processing failed: {e}", "timestamps": []}
    finally:
        if cap:
            cap.release()
        gc.collect()


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        try:
            if 'video' not in request.files:
                return render_template("result.html", prediction="Error: No file uploaded", video_path=None, details=None)

            file = request.files['video']
            if file.filename == '':
                return render_template("result.html", prediction="Error: No file selected", video_path=None, details=None)

            if not allowed_file(file.filename):
                return render_template("result.html", prediction="Error: Unsupported video type", video_path=None, details=None)

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                file.save(filepath)
                logger.info(f"File saved: {filepath}")
            except Exception as e:
                return render_template("result.html", prediction=f"Error saving file: {e}", video_path=None, details=None)

            logger.info("Starting video analysis...")
            start_time = time.time()
            result = predict_video_optimized(filepath)
            processing_time = time.time() - start_time
            logger.info(f"Video processed in {processing_time:.2f}s")

            if result.get("error"):
                return render_template('result.html', prediction=result["error"], video_path=None, details=None)

            # Don’t delete file immediately so it can be played in result.html
            return render_template('result.html', prediction=result["prediction"], video_path=filename, details=result)

        except Exception as e:
            logger.error(f"Upload error: {e}")
            traceback.print_exc()
            return render_template('result.html', prediction=f"Server error: {e}", video_path=None, details=None)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/camera_feed')
def camera_feed():
    return jsonify({"error": "Camera feed not available on Render"})


@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
