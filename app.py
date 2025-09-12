import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit for uploads

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model once
model = load_model("violence_detection_cnn.h5")
IMG_SIZE = (128, 128)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    """Resize and normalize image for model."""
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_video(video_path):
    """Predict violence in video using sampled frames."""
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        return f"Error reading video: {str(e)}"

    predictions = []
    duration = int(clip.duration)

    for t in range(0, duration, 1):  # 1 frame per second
        try:
            frame = clip.get_frame(t)
            frame_img = Image.fromarray(frame)
            frame_input = preprocess_image(frame_img)
            pred = model.predict(frame_input, verbose=0)[0][0]
            predictions.append(pred)
        except Exception as e:
            print("Frame processing error:", e)

    clip.close()

    if not predictions:
        return "Error: No frames could be processed"

    avg_pred = float(np.mean(predictions))
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

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Handle live camera frames from frontend."""
    try:
        data = request.get_json()
        image_data = data['image'].split(",")[1]  # remove data:image/...;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # preprocess + predict
        frame_input = preprocess_image(image)
        pred = model.predict(frame_input, verbose=0)[0][0]

        confidence = round(float(pred) * 100, 2)
        violence_detected = pred > 0.5
        threat_level = "High" if violence_detected else "Low"

        return jsonify({
            "violence_detected": violence_detected,
            "confidence": confidence,
            "threat_level": threat_level
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

