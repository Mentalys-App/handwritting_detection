import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size

# Load model
model = tf.keras.models.load_model('./model/mental_health_model.keras')

def prepare_image(image):
    """Prepare image for prediction"""
    try:
        # Read image in grayscale
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Unable to decode image")
        
        # Resize image
        img = cv2.resize(img, (128, 128))
        # Normalize and reshape
        img_array = np.expand_dims(img, axis=-1)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Error preparing image: {str(e)}")

def predict_image(image, temperature=1.5):
    """Make prediction with confidence calibration"""
    try:
        # Reset file pointer
        image.seek(0)
        
        # Prepare image
        img = prepare_image(image)
        img = np.expand_dims(img, axis=0)
        
        # Get raw prediction
        raw_prediction = model.predict(img, verbose=0)[0][0]
        
        # Apply temperature scaling for better calibration
        scaled_prediction = 1 / (1 + np.exp(-(np.log(raw_prediction / (1 - raw_prediction)) / temperature)))
        
        # Binary result - no intermediate uncertain state
        result = "Potential Mental Health Condition" if scaled_prediction > 0.5 else "No Mental Health Condition"
        
        return result, scaled_prediction
    except Exception as e:
        raise ValueError(f"Error making prediction: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and prediction"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return render_template('index.html', 
                                error='Invalid file type. Please upload an image file.')
        
        try:
            # Make prediction
            result, confidence = predict_image(file)
            
            # Prepare result message
            confidence_percentage = f"{confidence*100:.1f}%"
            
            return render_template('index.html',
                                result=result,
                                confidence=confidence_percentage,
                                filename=file.filename)
        except Exception as e:
            return render_template('index.html', 
                                error=f'Error processing image: {str(e)}')
    
    return render_template('index.html')

@app.errorhandler(413)
def too_large(e):
    return render_template('index.html', 
                         error='File is too large. Maximum size is 10MB'), 413

@app.errorhandler(Exception)
def handle_error(error):
    """Handle internal errors"""
    return render_template('index.html', error=str(error)), 500

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('./model/mental_health_model.keras'):
        raise FileNotFoundError("Model file not found. Please ensure the model is saved in the correct location.")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)