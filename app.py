from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
import tensorflow as tf
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure GPU memory growth to avoid memory allocation errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU mode active!")
    except RuntimeError as e:
        logger.error(e)

app = Flask(__name__)

# Set the base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'face_classifier.pkl')

# Load Face Recognition Model
try:
    logger.info(f"Trying to load model from {model_path}")
    classifier = joblib.load(model_path)
    embedder = FaceNet()
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Create a simple classifier to prevent app from crashing
    # This will return "Not Lia" for all inputs but at least the app will run
    from sklearn.svm import SVC
    classifier = SVC(probability=True)
    classifier.classes_ = np.array([0, 1])
    embedder = FaceNet()
    logger.warning("Using fallback classifier - app will work but recognition won't")

def preprocess_image(image):
    """Apply adaptive preprocessing to enhance face features"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized = clahe.apply(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return image  # Return original image if preprocessing fails

def detect_face(image, confidence_threshold=0.85):
    """Detect faces using Haar Cascade and classify with FaceNet+SVM model"""
    try:
        # Load face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.info("No faces detected in image")
            return False, 0
        
        # Process the largest face if multiple faces detected
        if len(faces) > 0:
            # Get largest face by area (width * height)
            x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
            face_img = image[y:y+h, x:x+w]
            
            # Resize to 160x160 (FaceNet input size)
            face_resized = cv2.resize(face_img, (160, 160))
            
            # Preprocess and get prediction
            img_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            img_preprocessed = preprocess_image(img_rgb)
            
            # Get face embedding
            embedding = embedder.embeddings([img_preprocessed])[0]
            
            # Get prediction probability
            try:
                prob = classifier.predict_proba([embedding])[0][1]
                
                # Classify based on threshold
                is_lia = prob > confidence_threshold
                
                logger.info(f"Face detected with confidence: {prob:.2f}")
                return is_lia, prob
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                return False, 0
        
        return False, 0
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return False, 0

@app.route('/')
def index():
    """Render the main page"""
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    """Process uploaded image and return prediction"""
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        is_lia, confidence = detect_face(image)

        response = {
            "result": "Lia Detected" if is_lia else "Not Lia",
            "confidence": float(confidence),
            "message": "Hei! If you see this message it means you are Lia! I'm Rangga face recognition assistant, it's nice to see you! Please kindly tell Rangga if you already see this messages! He has something need to say hehehe~ Good luck for you both!" if is_lia else "Sorry, this is not Lia!"
        }
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({"error": str(e), "result": "Error", "message": "An error occurred while processing your image. Please try again."}), 500

if __name__ == '__main__':
    # Get port from environment variable (Heroku sets this)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)