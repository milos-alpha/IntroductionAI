import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, TOP, BOTTOM, X, LEFT, RIGHT, StringVar, ttk
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
import os
import sys

# Import the preprocessing functions from traffic.py
try:
    from traffic import preprocess_image, load_trained_model
except ImportError:
    print("Warning: Could not import functions from traffic.py. Using built-in versions.")
    
    def preprocess_image(image_path):
        """
        Preprocess a single image for prediction.
        """
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (30, 30))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def load_trained_model(model_path):
        """
        Load a trained model from file.
        """
        try:
            model = tf.keras.models.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Dictionary mapping category numbers to traffic sign names
SIGN_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}


def predict_sign(image_path, model_path="best_model.h5"):
    """
    Takes an image path as argument and returns a prediction
    (a number, its corresponding sign, and the probability of correctness)
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model file
            
    Returns:
        tuple: (category_number, sign_name, probability)
    """
    # Load model
    model = load_trained_model(model_path)
    if model is None:
        return (-1, "Error loading model", 0.0)
    
    # Preprocess image
    img = preprocess_image(image_path)
    if img is None:
        return (-1, "Error processing image", 0.0)
    
    # Normalize pixel values
    img = img / 255.0
    
    # Make prediction
    predictions = model.predict(img, verbose=0)
    category = np.argmax(predictions[0])
    probability = np.max(predictions[0])
    
    # Get sign name from category
    sign_name = SIGN_NAMES.get(category, "Unknown Sign")
    
    return category, sign_name, probability


class TrafficSignApp:
    def __init__(self, root, model_path="best_model.h5"):
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("800x600")
        self.model_path = model_path
        
        # Create styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10), background="#f0f0f0")
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), background="#f0f0f0")
        self.style.configure("Result.TLabel", font=("Arial", 12), background="#f0f0f0")
        
        # Main container
        self.main_frame = ttk.Frame(root, style="TFrame", padding=10)
        self.main_frame.pack(fill="both", expand=True)
        
        # Header
        self.header_label = ttk.Label(
            self.main_frame, 
            text="Traffic Sign Recognition", 
            style="Header.TLabel"
        )
        self.header_label.pack(pady=10)
        
        # Button frame
        self.button_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.button_frame.pack(fill="x", pady=10)
        
        self.upload_button = ttk.Button(
            self.button_frame, 
            text="Upload Image", 
            command=self.upload_image
        )
        self.upload_button.pack(side="left", padx=5)
        
        self.predict_button = ttk.Button(
            self.button_frame, 
            text="Predict Sign", 
            command=self.predict_sign_from_ui
        )
        self.predict_button.pack(side="right", padx=5)
        
        # Image display frame
        self.image_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.image_frame.pack(fill="both", expand=True, pady=10)
        
        self.image_label = ttk.Label(
            self.image_frame, 
            text="No image uploaded", 
            style="TLabel"
        )
        self.image_label.pack(pady=10)
        
        # Result frame
        self.result_frame = ttk.Frame(self.main_frame, style="TFrame")
        self.result_frame.pack(fill="x", pady=10)
        
        self.result_var = StringVar()
        self.result_var.set("")
        
        self.result_label = ttk.Label(
            self.result_frame, 
            textvariable=self.result_var, 
            style="Result.TLabel"
        )
        self.result_label.pack(pady=10)
        
        # Status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        
        self.status_bar = ttk.Label(
            root, 
            textvariable=self.status_var, 
            relief="sunken", 
            anchor="w"
        )
        self.status_bar.pack(side="bottom", fill="x")
        
        self.image_path = None
        self.tk_image = None
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            self.status_var.set(f"Warning: Model file '{self.model_path}' not found. Please train the model first.")
            self.predict_button.state(["disabled"])
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Traffic Sign Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            self.result_var.set("Click 'Predict Sign' to analyze the image")
    
    def display_image(self, image_path):
        try:
            # Open image with PIL
            pil_image = Image.open(image_path)
            
            # Resize image to fit in the window (maintaining aspect ratio)
            max_width, max_height = 400, 400
            width, height = pil_image.size
            
            if width > height:
                new_width = max_width
                new_height = int(height * (max_width / width))
            else:
                new_height = max_height
                new_width = int(width * (max_height / height))
            
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to Tkinter PhotoImage
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=self.tk_image, text="")
            self.image_label.image = self.tk_image
        except Exception as e:
            self.status_var.set(f"Error displaying image: {str(e)}")
    
    def predict_sign_from_ui(self):
        if self.image_path:
            self.status_var.set("Analyzing image...")
            self.root.update()
            
            try:
                category, sign_name, probability = predict_sign(self.image_path, self.model_path)
                
                if category == -1:
                    self.result_var.set(f"Error: {sign_name}")
                    self.status_var.set("Prediction failed")
                else:
                    result_text = f"Prediction: Category {category}\nSign: {sign_name}\nProbability: {probability:.4f}"
                    self.result_var.set(result_text)
                    self.status_var.set("Prediction complete")
            except Exception as e:
                self.result_var.set(f"Error during prediction: {str(e)}")
                self.status_var.set("Prediction failed")
        else:
            self.status_var.set("No image loaded")
            self.result_var.set("Please upload an image first")


def main():
    # Parse command line arguments for model path
    model_path = "best_model.h5"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Initialize the application
    root = tk.Tk()
    app = TrafficSignApp(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()