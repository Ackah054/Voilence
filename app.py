import os
import cv2
import numpy as np
import logging
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# ------------------------------
# Flask app configuration
# ------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit: 50 MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Load trained model
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)


# ------------------------------
# Video prediction function
# ------------------------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    max_frames = 300  # Avoid timeout on Render (process up to 300 frames)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
            break

        if frame_count % 10 == 0:  # Sample every 10th frame
            try:
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)
                pred = model.predict(frame_input, verbose=0)[0][0]
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Frame processing error: {e}")
        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return "Error: No frames could be processed"

    avg_pred = np.mean(predictions)
    return "Violence" if avg_pred > 0.5 else "No Violence"


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file uploaded", 400

        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure folder exists before saving
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        try:
            result = predict_video(filepath)
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return f"Error during analysis: {str(e)}", 500

        return render_template('result.html', prediction=result, video_path=filepath)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


# ------------------------------
# Main entry
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
"""










import os
import cv2
import numpy as np
import logging
import base64
import io
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image

# ------------------------------
# Flask app configuration
# ------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit: 50 MB

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Load trained model
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)


# ------------------------------
# Video prediction function
# ------------------------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0
    max_frames = 300  # Avoid timeout on Render (process up to 300 frames)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
            break

        if frame_count % 10 == 0:  # Sample every 10th frame
            try:
                frame_resized = cv2.resize(frame, IMG_SIZE)
                frame_norm = frame_resized / 255.0
                frame_input = np.expand_dims(frame_norm, axis=0)
                pred = model.predict(frame_input, verbose=0)[0][0]
                predictions.append(pred)
            except Exception as e:
                logging.error(f"Frame processing error: {e}")
        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return "Error: No frames could be processed"

    avg_pred = np.mean(predictions)
    return "Violence" if avg_pred > 0.5 else "No Violence"


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return "No file uploaded", 400

        file = request.files['video']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Ensure folder exists before saving
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        try:
            result = predict_video(filepath)
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return f"Error during analysis: {str(e)}", 500

        return render_template('result.html', prediction=result, video_path=filepath)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    try:
        # Get the base64 image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess frame for model
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_norm = frame_resized / 255.0
        frame_input = np.expand_dims(frame_norm, axis=0)
        
        # Make prediction
        prediction = model.predict(frame_input, verbose=0)[0][0]
        
        # Determine result
        is_violence = prediction > 0.5
        confidence = float(prediction) if is_violence else float(1 - prediction)
        
        return jsonify({
            'violence_detected': is_violence,
            'confidence': round(confidence * 100, 2),
            'prediction_score': float(prediction),
            'status': 'Violence Detected' if is_violence else 'Safe',
            'threat_level': 'High' if is_violence else 'Low'
        })
        
    except Exception as e:
        logging.error(f"Error in real-time detection: {e}")
        return jsonify({'error': str(e)}), 500


# ------------------------------
# Main entry
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
"""