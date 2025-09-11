import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"

# Ensure the upload folder exists (important for Render)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)

def predict_video(video_path):
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
                print("Frame processing error:", e)
        frame_count += 1
    cap.release()

    if len(predictions) == 0:
        return "Error: No frames could be processed"

    avg_pred = np.mean(predictions)
    return "Violence" if avg_pred > 0.5 else "No Violence"

# Routes
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

        # Ensure folder exists each request
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        file.save(filepath)

        try:
            result = predict_video(filepath)
        except Exception as e:
            return f"Error during analysis: {str(e)}", 500

        return render_template('result.html', prediction=result, video_path=filepath)

    return render_template('upload.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)
