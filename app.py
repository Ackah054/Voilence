import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"

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
        if frame_count % 10 == 0:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_norm = frame_resized / 255.0
            frame_input = np.expand_dims(frame_norm, axis=0)
            pred = model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)
        frame_count += 1
    cap.release()
    
    avg_pred = np.mean(predictions)
    return "Violence" if avg_pred > 0.5 else "No Violence"

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files['video']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predict_video(filepath)
        return render_template('result.html', prediction=result, video_path=filepath)
    return render_template('upload.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
