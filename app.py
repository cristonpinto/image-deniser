from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import uuid
import numpy as np
import cv2
from PIL import Image
import time
from werkzeug.utils import secure_filename
from utils.denoiser import load_model, add_noise_tf, denoise_image, denoise_image_patches, calculate_metrics

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the model
model = load_model(app.config['MODEL_PATH'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info("Upload endpoint called")
    
    if 'file' not in request.files:
        app.logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        app.logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filenames
        unique_id = str(uuid.uuid4())
        original_filename = secure_filename(f"{unique_id}_original.png")
        
        try:
            # Ensure directories exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save the uploaded image
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(original_path)
            
            app.logger.info(f"File saved to {original_path}")
            
            # Return the image ID for further processing
            return jsonify({
                'success': True, 
                'image_id': unique_id,
                'original_url': url_for('uploaded_file', filename=original_filename)
            })
        
        except Exception as e:
            app.logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/denoise', methods=['POST'])
def denoise_image_endpoint():
    data = request.json
    if not data or 'image_id' not in data:
        return jsonify({'error': 'No image ID provided'}), 400
    
    image_id = data['image_id']
    noise_type = data.get('noise_type', 'gaussian')
    noise_level = float(data.get('noise_level', 0.1))
    use_patches = data.get('use_patches', True)
    patch_size = int(data.get('patch_size', 128))
    patch_overlap = int(data.get('patch_overlap', 16))
    
    # Find the original image
    original_filename = f"{image_id}_original.png"
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    
    if not os.path.exists(original_path):
        return jsonify({'error': 'Original image not found'}), 404
    
    try:
        # Read the image
        img = cv2.imread(original_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if necessary (when not using patches)
        if not use_patches:
            img = cv2.resize(img, (128, 128))
        
        # Convert to float32 and normalize
        original_img = img.astype('float32') / 255.0
        
        # Add noise
        noisy_img = add_noise_tf(original_img, noise_type=noise_type, noise_level=noise_level).numpy()
        
        # Create filenames for noisy and denoised images
        noisy_filename = f"{image_id}_noisy.png"
        denoised_filename = f"{image_id}_denoised.png"
        noisy_path = os.path.join(app.config['RESULT_FOLDER'], noisy_filename)
        denoised_path = os.path.join(app.config['RESULT_FOLDER'], denoised_filename)
        
        # Save noisy image
        save_image(noisy_img, noisy_path)
        
        # Denoise
        start_time = time.time()
        if use_patches:
            denoised_img = denoise_image_patches(model, noisy_img, patch_size=patch_size, overlap=patch_overlap)
        else:
            denoised_img = denoise_image(model, noisy_img)
        processing_time = time.time() - start_time
        
        # Save denoised image
        save_image(denoised_img, denoised_path)
        
        # Calculate metrics
        psnr_val, ssim_val = calculate_metrics(original_img, denoised_img)
        
        return jsonify({
            'success': True,
            'noisy_url': url_for('result_file', filename=noisy_filename),
            'denoised_url': url_for('result_file', filename=denoised_filename),
            'metrics': {
                'psnr': float(psnr_val),
                'ssim': float(ssim_val),
                'processing_time': processing_time
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_image(img_array, path):
    # Convert from float32 [0,1] to uint8 [0,255]
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(path)

if __name__ == '__main__':
    app.run(debug=True)