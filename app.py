import os
import cv2
import numpy as np
import base64
import io
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from skimage import metrics

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 1. HELPER FUNCTIONS (Restored Full Logic) ---

def calculate_psnr(original, compressed):
    """Menghitung Peak Signal-to-Noise Ratio (PSNR)."""
    return metrics.peak_signal_noise_ratio(original, compressed, data_range=255)

def calculate_mse(original, compressed):
    """Menghitung Mean Squared Error (MSE)."""
    return metrics.mean_squared_error(original, compressed)

def add_salt_and_pepper_noise(image, density):
    noisy_image = image.copy()
    num_pixels = image.size
    num_salt = np.ceil(density * num_pixels * 0.5)
    num_pepper = np.ceil(density * num_pixels * 0.5)
    
    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255
    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    return noisy_image

def box_filter(image, kernel_size=3):
    """Filter Weighted Average (Custom 3x3 Kernel)."""
    # Define a standard 3x3 weighted kernel (approx Gaussian)
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    
    # Normalize: Divide by sum of weights (16) so image doesn't get brighter
    kernel = kernel / 16.0 
    
    # Apply the specific kernel
    return cv2.filter2D(image, -1, kernel)

def standard_median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def simple_adaptive_median_filter(image, kernel_size=3):
    median_filtered = cv2.medianBlur(image, kernel_size)
    noise_mask = (image == 0) | (image == 255)
    restored_image = np.where(noise_mask, median_filtered, image)
    return restored_image.astype(np.uint8)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

# --- 2. MAIN ROUTE ---

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
            
        if file and allowed_file(file.filename):
            # Read image directly from memory
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            original_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if original_img is None:
                return render_template('index.html', error='Invalid image file')
            
            original_img = cv2.resize(original_img, (512, 512))
            density = float(request.form.get('density', 0.3))
            
            # B. EXECUTE FILTERS
            noisy_img = add_salt_and_pepper_noise(original_img, density)
            
            res_box = box_filter(noisy_img)
            res_smf = standard_median_filter(noisy_img)
            res_amf = simple_adaptive_median_filter(noisy_img)
            
            # C. CALCULATE METRICS
            stats = {
                'box': {
                    'psnr': round(calculate_psnr(original_img, res_box), 2),
                    'mse': round(calculate_mse(original_img, res_box), 2)
                },
                'smf': {
                    'psnr': round(calculate_psnr(original_img, res_smf), 2),
                    'mse': round(calculate_mse(original_img, res_smf), 2)
                },
                'amf': {
                    'psnr': round(calculate_psnr(original_img, res_amf), 2),
                    'mse': round(calculate_mse(original_img, res_amf), 2)
                }
            }

            # D. CONVERT IMAGES TO BASE64
            base_name = os.path.splitext(secure_filename(file.filename))[0]
            images_data = {
                'original': {
                    'data': image_to_base64(original_img),
                    'filename': f"{base_name}_original.png"
                },
                'noisy': {
                    'data': image_to_base64(noisy_img),
                    'filename': f"{base_name}_noisy.png"
                },
                'box': {
                    'data': image_to_base64(res_box),
                    'filename': f"{base_name}_box_filter.png"
                },
                'smf': {
                    'data': image_to_base64(res_smf),
                    'filename': f"{base_name}_standard_median.png"
                },
                'amf': {
                    'data': image_to_base64(res_amf),
                    'filename': f"{base_name}_adaptive_median.png"
                }
            }

            # Return JSON response with images and stats
            return jsonify({
                'success': True,
                'images': images_data,
                'stats': stats,
                'density': density
            })

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)