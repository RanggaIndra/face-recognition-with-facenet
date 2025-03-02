from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
from keras_facenet import FaceNet
import tensorflow as tf
import os

# Configure GPU memory growth to avoid memory allocation errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU mode active!")
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

# Load Face Recognition Model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'face_classifier.pkl')
classifier = joblib.load(model_path)
embedder = FaceNet()

def preprocess_image(image):
    """Apply adaptive preprocessing to enhance face features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def detect_face(image, confidence_threshold=0.85):
    """Detect faces using Haar Cascade and classify with FaceNet+SVM model"""
    # Load face cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
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
        prob = classifier.predict_proba([embedding])[0][1]
        
        # Classify based on threshold
        is_lia = prob > confidence_threshold
        
        return is_lia, prob
    
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
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)