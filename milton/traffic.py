# traffic_model.py
import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Iterate through each subdirectory (category)
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        
        # Check if the category directory exists
        if not os.path.exists(category_path):
            continue
        
        # Iterate through image files in the category directory
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            
            # Read image and resize to specified dimensions
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # Ensure the image is in the correct color space (BGR to RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                images.append(img)
                labels.append(category)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten layer to transition from 2D to 1D
        tf.keras.layers.Flatten(),
        
        # Dense layers
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(data_dir, model_output=None):
    """
    Train the model and save it to a file.
    
    Args:
        data_dir: Directory containing the training data
        model_output: Optional path to save the model
        
    Returns:
        Trained model
    """
    # Get image arrays and labels for all image files
    images, labels = load_data(data_dir)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model to file
    if model_output:
        model.save(model_output)
        print(f"Model saved to {model_output}.")
    
    return model


def preprocess_image(image_path):
    """
    Preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Read and preprocess the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def load_trained_model(model_path):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None