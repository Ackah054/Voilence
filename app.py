import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import traceback

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increased to 100 MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model once
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)


def allowed_file(filename):
    """Check if file extension is allowed (.mp4 only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_video(video_path):
    """Predict violence in a saved video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video file"

    predictions = []
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = frame_count // fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return "Error: Video contains no readable frames"

    # Sample 1 frame per second
    for t in range(duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)  # jump to t-th second
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_norm = frame_resized.astype(np.float32) / 255.0
            frame_input = np.expand_dims(frame_norm, axis=0)
            pred = model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)
        except Exception as e:
            print("Frame processing error:", e)

    cap.release()

    if len(predictions) == 0:
        return "Error: No frames could be processed"

    avg_pred = float(np.mean(predictions))
    return "Violence" if avg_pred > 0.5 else "No Violence"


def gen_camera():
    """Real-time camera feed with violence detection."""
    cap = cv2.VideoCapture(0)  # system webcam (0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_input = np.expand_dims(frame_norm, axis=0)
        pred = model.predict(frame_input, verbose=0)[0][0]

        label = "Violence" if pred > 0.5 else "No Violence"
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

        # Draw label on frame
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template("result.html", prediction="Error: No file uploaded", video_path=None)

        file = request.files['video']
        if file.filename == '':
            return render_template("result.html", prediction="Error: No file selected", video_path=None)

        if not allowed_file(file.filename):
            return render_template("result.html", prediction="Error: Only .mp4 files are allowed", video_path=None)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_video(filepath)
        except Exception as e:
            traceback.print_exc()  # print error in Render logs
            result = f"Error during analysis: {str(e)}"

        return render_template('result.html', prediction=result, video_path=filename)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    """Camera page template."""
    return render_template('camera.html')


@app.route('/camera_feed')
def camera_feed():
    """Live camera feed with predictions."""
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
