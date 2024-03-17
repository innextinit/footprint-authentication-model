import os
from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from database_operations import save_features, find_matching_features
from train_model import load_model

current_directory = os.getcwd()
static_dir = os.path.join(current_directory, 'static')

app = Flask(__name__)

#Transforms
transformer = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize(mean=[0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        std=[0.5,0.5,0.5])
])

def extract_features(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transformed_image = transformer(image)
    processed_image = torch.unsqueeze(transformed_image, 0)

    # return processed_image

    trained_model = load_model("trained_model.pth")
    
    with torch.no_grad():
        features = trained_model(processed_image)
        weight, _ = torch.max(features, 1)

    return weight.item(), processed_image.numpy()

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
    
    predicted_weight, processed_image = extract_features(uploaded_file)

    # Save features to the database
    save_features(uploaded_file.filename, predicted_weight, processed_image)

    return jsonify({'message': f'Footprint {uploaded_file.filename} registered successfully'}), 200


@app.route('/access', methods=['POST'])
def authenticate_footprint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    predicted_weight, processed_image = extract_features(uploaded_file)

    # Find matching features in the database
    matched_results = find_matching_features(uploaded_file.filename, predicted_weight, processed_image)

    return jsonify({'results': matched_results}), 200

if __name__ == '__main__':
    app.run(debug=True)
