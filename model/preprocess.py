import cv2
import numpy as np

def preprocess_image(image):
    """
    Adaptive preprocessing to enhance face features for better recognition.
    
    Parameters:
    image (numpy.ndarray): Input RGB image
    
    Returns:
    numpy.ndarray: Preprocessed RGB image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    # Convert back to RGB
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def extract_face(image, detector):
    """
    Extract face from image using the detector.
    
    Parameters:
    image (numpy.ndarray): Input BGR image
    detector: Face detector (MTCNN or cascade)
    
    Returns:
    numpy.ndarray: Cropped and preprocessed face (160x160)
    """
    # Convert to RGB for MTCNN
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if hasattr(detector, 'detect_faces'):  # MTCNN detector
        faces = detector.detect_faces(img_rgb)
        if not faces:
            return None
            
        # Get face box
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)  # Ensure non-negative
        face = img_rgb[y:y+height, x:x+width]
    else:  # Cascade detector
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
            
        # Get largest face
        x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
        face = img_rgb[y:y+h, x:x+w]
    
    # Resize to FaceNet input size
    face_resized = cv2.resize(face, (160, 160))
    
    # Apply additional preprocessing
    face_preprocessed = preprocess_image(face_resized)
    
    return face_preprocessed

def augment_image(image):
    """
    Apply data augmentation to create variations of the input image.
    
    Parameters:
    image (numpy.ndarray): Input RGB image
    
    Returns:
    list: List of augmented images
    """
    augmented_images = []

    # Flip horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    # Rotation
    for angle in [-15, -10, 10, 15]:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    # Brightness variation (Gamma Correction)
    for gamma in [0.3, 0.5, 0.7]:  # Lower gamma = darker image
        invGamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        brightness_variation = cv2.LUT(image, table)
        augmented_images.append(brightness_variation)

    return augmented_images