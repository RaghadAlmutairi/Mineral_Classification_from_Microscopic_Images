from flask import Flask, request, jsonify, render_template 
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
model_path = os.path.abspath('rock_classification_model2.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = load_model(model_path)
#model.summary()

# Check input shape from the model
input_shape = model.input_shape  # Example: (None, 148, 148, 3)
IMG_SIZE = (input_shape[1], input_shape[2])  # Extract correct size
COLOR_MODE = "rgb" if input_shape[3] == 3 else "grayscale"  # Check channels

# Define class labels (Ensure correct number)
CLASS_LABELS = [ "Anorthosite", "Augite", "Garnet", "Hornblende", "Olivine", "Oolite", "Staurolite","Biotite" , "Chlorite","Calcite" , "Microcline" , "Muscovite", "Plagioclase"]
  # Ensure exactly 9 labels

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Load image with correct mode and size
            img = image.load_img(filepath, target_size=IMG_SIZE, color_mode=COLOR_MODE)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0
            
            print(f"Preprocessed image shape: {img_array.shape}")

            # Predict class
            prediction = model.predict(img_array)
            print(f"Model prediction output: {prediction}")
            class_index = np.argmax(prediction, axis=1)[0]
            
            if class_index >= len(CLASS_LABELS):
                print(f"Error: Index {class_index} is out of bounds for CLASS_LABELS.")
                return jsonify({'error': 'Unexpected classification index'}), 500

            class_label = CLASS_LABELS[class_index]
            confidence = float(np.max(prediction))

            return jsonify({'class': class_label, 'confidence': confidence})

        except Exception as e:
            print(f"Prediction Error: {str(e)}")  # Log error for debugging
            return jsonify({'error': 'Classification failed'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
