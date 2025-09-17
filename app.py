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
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'wmv'}  # allow more formats
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model once
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_video(video_path, sample_rate=5, threshold=0.5):
    """
    Predict violence in a saved video file using OpenCV.
    Returns overall result + list of timestamps where violence detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"overall": "Error: Could not open video file", "timestamps": []}

    predictions = []
    violence_times = []

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = frame_count // fps if fps > 0 else 0

    if duration == 0:
        cap.release()
        return {"overall": "Error: Video contains no readable frames", "timestamps": []}

    # Sample frames every `sample_rate` seconds
    for t in range(0, duration, sample_rate):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_norm = frame_resized.astype(np.float32) / 255.0
            frame_input = np.expand_dims(frame_norm, axis=0)
            pred = model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)

            if pred > threshold:
                violence_times.append(t)  # seconds
        except Exception as e:
            print("Frame processing error:", e)

    cap.release()

    if len(predictions) == 0:
        return {"overall": "Error: No frames could be processed", "timestamps": []}

    avg_pred = float(np.mean(predictions))
    overall = "Violence" if avg_pred > threshold else "No Violence"

    return {"overall": overall, "timestamps": violence_times}


def gen_camera():
    """Real-time camera feed with violence detection."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_norm = frame_resized.astype(np.float32) / 255.0
        frame_input = np.expand_dims(frame_norm, axis=0)
        pred = model.predict(frame_input, verbose=0)[0][0]

        label = "Violence" if pred > 0.5 else "No Violence"
        color = (0, 0, 255) if label == "Violence" else (0, 255, 0)

        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

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
            return render_template("result.html",
                                   prediction="Error: No file uploaded",
                                   video_path=None,
                                   timestamps=[])

        file = request.files['video']
        if file.filename == '':
            return render_template("result.html",
                                   prediction="Error: No file selected",
                                   video_path=None,
                                   timestamps=[])

        if not allowed_file(file.filename):
            return render_template("result.html",
                                   prediction="Error: Unsupported file format",
                                   video_path=None,
                                   timestamps=[])

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            result = predict_video(filepath)
            prediction = result["overall"]
            timestamps = result["timestamps"]
        except Exception as e:
            traceback.print_exc()
            prediction = f"Error during analysis: {str(e)}"
            timestamps = []

        return render_template('result.html',
                               prediction=prediction,
                               video_path=filename,
                               timestamps=timestamps)

    return render_template('upload.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/camera_feed')
def camera_feed():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # use Render's port, fallback 5000 for local
    app.run(host="0.0.0.0", port=port, debug=True)

