# predict.py
import sys
import os
import cv2
import numpy as np
from traffic import load_trained_model, preprocess_image
from sign_names import SIGN_NAMES

def predict_traffic_sign(model_path, image_path):
    """
    Predict the class of a traffic sign in an image.
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image file
        
    Returns:
        Tuple of (predicted class index, class name, confidence)
    """
    # Load the trained model
    model = load_trained_model(model_path)
    if model is None:
        return None, None, None
    
    # Preprocess the image
    img = preprocess_image(image_path)
    if img is None:
        return None, None, None
    
    # Make a prediction
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    # Get the class name
    class_name = SIGN_NAMES.get(class_idx, f"Unknown (Class {class_idx})")
    
    return class_idx, class_name, confidence


def main():
    """
    Predict the class of a traffic sign in an image.
    
    Usage: python predict.py model.h5 image_file
    """
    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python predict.py model.h5 image_file")
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.isfile(model_path):
        sys.exit(f"Error: Model file '{model_path}' does not exist")
    
    if not os.path.isfile(image_path):
        sys.exit(f"Error: Image file '{image_path}' does not exist")
    
    # Predict the traffic sign
    class_idx, class_name, confidence = predict_traffic_sign(model_path, image_path)
    
    if class_idx is not None:
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("Failed to make a prediction.")


if __name__ == "__main__":
    main()