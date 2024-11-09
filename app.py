from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('brain_clot_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def process_image(filepath):
    # Load and preprocess image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    
    # Make prediction
    input_arr = np.expand_dims(img_array, axis=0)
    prediction = model.predict(input_arr)[0][0]
    
    # Apply Sobel filters
    img_gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.sqrt(sobelx**2 + sobely**2)
    
    # Save filtered images
    def normalize_and_save(img, filename):
        img_normalized = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(save_path, img_normalized)
        return save_path

    sobelx_path = normalize_and_save(sobelx, 'sobelx.jpg')
    sobely_path = normalize_and_save(sobely, 'sobely.jpg')
    combined_path = normalize_and_save(sobel_combined, 'combined.jpg')
    
    return {
        'prediction': float(prediction),
        'has_clot': bool(prediction > 0.5),
        'sobelx_path': sobelx_path,
        'sobely_path': sobely_path,
        'combined_path': combined_path
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = process_image(filepath)
            results['original_path'] = filepath
            return jsonify(results)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=9000, debug=True)