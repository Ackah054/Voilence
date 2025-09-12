import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit for uploads

# Ensure the upload folder exists (important for Render)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model once
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)

def allowed_file(filename):
    """Check if file extension is allowed (.mp4 only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_video(video_path):
    """Predict violence in video using sampled frames."""
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        return f"Error reading video: {str(e)}"

    predictions = []
    duration = int(clip.duration)

    # Sample 1 frame per second (instead of all frames)
    for t in range(0, duration, 1):
        try:
            frame = clip.get_frame(t)  # numpy array (H, W, 3)
            frame_resized = np.array(
                np.resize(frame, (*IMG_SIZE, 3)), dtype=np.float32
            )
            frame_norm = frame_resized / 255.0
            frame_input = np.expand_dims(frame_norm, axis=0)
            pred = model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)
        except Exception as e:
            print("Frame processing error:", e)

    clip.close()

    if len(predictions) == 0:
        return "Error: No frames could be processed"

    avg_pred = float(np.mean(predictions))
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

        if not allowed_file(file.filename):
            return "Only .mp4 files are allowed", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
