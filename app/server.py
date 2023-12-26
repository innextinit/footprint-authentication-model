import os
from flask import Flask, request, jsonify, send_file
import keras
from keras.models import load_model
from image_processing import preprocess_image
from database_operations import save_features, find_matching_features

current_directory = os.getcwd()
static_dir = os.path.join(current_directory, 'static')

app = Flask(__name__)

# Load the trained saved CNN model
model = load_model(os.path.join("saved_models", "footprint_auth_model.h5"))

# Create a new model to extract features from an intermediate layer
feature_extraction_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

# Freeze the layers to prevent training
for layer in feature_extraction_model.layers:
    layer.trainable = False

@app.route('/')
def index():
    return send_file(os.path.join(static_dir, "index.html"))

@app.route('/reg', methods=['POST'])
def register_footprint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # temp store the file and ref the path to the file
    file_path = os.path.join("temp", uploaded_file.filename)
    uploaded_file.save(file_path)

    # Preprocess the uploaded image
    processed_image = preprocess_image(file_path)
    
      # Extract features from the preprocessed image using the trained CNN model
    extracted_features = feature_extraction_model.predict(processed_image.reshape(1, 150, 150, 3))  # Assuming image size is 150x150

    image_name = uploaded_file.filename
    save_features(image_name, extracted_features)

    return jsonify({'message': f'Footprint {image_name} registered successfully'}), 200

@app.route('/access', methods=['POST'])
def authenticate_footprint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # temp store the file and ref the path to the file
    file_path = os.path.join("temp", uploaded_file.filename)
    uploaded_file.save(file_path)

    # Preprocess the uploaded image
    processed_image = preprocess_image(file_path)
    
    # Extract features from the preprocessed image using the trained CNN model
    extracted_features = feature_extraction_model.predict(processed_image.reshape(1, 150, 150, 3))  # Assuming image size is 150x150


    # Find matching features in the database
    # Replace 'extracted_features' with the actual features extracted
    matched_results = find_matching_features(extracted_features)

    # Return matching results or relevant information
    return jsonify({'results': matched_results}), 200

if __name__ == '__main__':
    app.run(debug=True)
