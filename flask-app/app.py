from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Paths to both models
EFFICIENTNET_MODEL_PATH = os.path.join("models", "Efficient_classify_v2b3.keras")
MOBILENET_MODEL_PATH = os.path.join("models", "Mobile_classify_v3large.keras")

# Load both models
efficientnet_model = load_model(EFFICIENTNET_MODEL_PATH)
mobilenet_model = load_model(MOBILENET_MODEL_PATH)

# Common class labels
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({'error': 'Missing file or model type'}), 400

    file = request.files['file']
    model_type = request.form['model']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(file).convert('RGB')

        # Resize based on model
        if model_type == 'efficientnet':
            model = efficientnet_model
            target_size = (300, 300)
        elif model_type == 'mobilenet':
            model = mobilenet_model
            target_size = (128, 128)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = float(predictions[0][predicted_index])

        return jsonify({
            'model_used': model_type,
            'class': class_names[predicted_index],
            'confidence': f"{confidence * 100:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
