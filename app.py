import os
import json
import secrets
import numpy as np
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static', 'images')
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenetv2_best.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')
IMG_SIZE = (224, 224)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load model and class names globally
model = None
class_names = []

def load_model_and_classes():
    global model, class_names
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded {len(class_names)} class names")
    except Exception as e:
        print(f"Error loading class names: {e}")
        # Fallback to hardcoded list
        class_names = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
            "Apple___healthy", "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy", "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]

def preprocess_image(filepath):
    """Preprocess image for MobileNetV2 prediction"""
    with Image.open(filepath) as img:
        img_rgb = img.convert('RGB')
        img_resized = img_rgb.resize(IMG_SIZE)
        img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    return img_array

def parse_class_name(raw_class):
    """Parse raw class name into plant type and condition"""
    parts = raw_class.split('___')
    plant = parts[0].replace('_', ' ').replace(',', '')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    is_healthy = 'healthy' in condition.lower()
    return plant, condition, is_healthy

def predict_image(filepath):
    """Run inference and return prediction dict"""
    if model is None:
        raise ValueError("Model not loaded")

    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100

    raw_class = class_names[predicted_idx] if predicted_idx < len(class_names) else "Unknown"
    plant_type, condition, is_healthy = parse_class_name(raw_class)

    # Generate recommendations
    if is_healthy:
        recommendations = [
            "Continue regular watering and care",
            "Ensure adequate sunlight and nutrients",
            "Monitor for any changes in appearance",
            "Maintain good air circulation"
        ]
    else:
        recommendations = [
            "Isolate affected plants to prevent spread",
            "Consult with an agricultural expert",
            "Consider appropriate treatment methods",
            "Monitor other plants for similar symptoms"
        ]

    return {
        'raw_class': raw_class,
        'plant_type': plant_type,
        'condition': condition,
        'is_healthy': is_healthy,
        'confidence': round(confidence, 2),
        'recommendations': recommendations
    }

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secrets.token_hex(8) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            prediction = predict_image(filepath)

            # Save image to static folder for display
            static_filename = f"upload_{secrets.token_hex(8)}.jpg"
            static_path = os.path.join(app.config['STATIC_FOLDER'], static_filename)
            with Image.open(filepath) as img:
                img.convert('RGB').save(static_path)

            # Store in session
            session['prediction'] = prediction
            session['image_path'] = f'images/{static_filename}'

            os.remove(filepath)  # Clean up temp file
            return jsonify({'success': True})

        except Exception as e:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    prediction = session.get('prediction')
    image_path = session.get('image_path')
    if not prediction:
        return redirect(url_for('upload'))
    return render_template('result.html', prediction=prediction, image_path=image_path)

# Initialize application dependencies
load_model_and_classes()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
