import os
import cv2
import numpy as np
import logging
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import moviepy.editor as mp

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup (important for debugging in Render)
logging.basicConfig(level=logging.INFO)

# Load trained model
MODEL_PATH = "violence_detection_cnn.h5"
model = load_model(MODEL_PATH)
IMG_SIZE = (128, 128)


def predict_video(video_path):
    """Process video frames and predict violence or not."""
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
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
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        # Convert AVI to MP4 for browser compatibility
        if filepath.lower().endswith(".avi"):
            mp4_path = filepath.rsplit(".", 1)[0] + ".mp4"
            try:
                clip = mp.VideoFileClip(filepath)
                clip.write_videofile(mp4_path, codec="libx264")
                filepath = mp4_path
                logging.info(f"Converted AVI to MP4: {filepath}")
            except Exception as e:
                logging.error(f"Error converting video: {e}")
                return f"Error converting video: {str(e)}", 500

        try:
            result = predict_video(filepath)
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            return f"Error during analysis: {str(e)}", 500

        return render_template(
            'result.html',
            prediction=result,
            video_path=filepath
        )

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)
