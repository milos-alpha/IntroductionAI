# traffic_sign_app.py
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from traffic import load_trained_model, preprocess_image
from sign_names import SIGN_NAMES

class TrafficSignApp:
    def __init__(self, root, model_path):
        """
        Initialize the Traffic Sign Recognition App.
        
        Args:
            root: Tkinter root window
            model_path: Path to the trained model
        """
        self.root = root
        self.root.title("Traffic Sign Recognition")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Load the model
        self.model = load_trained_model(model_path)
        if self.model is None:
            messagebox.showerror("Error", f"Failed to load model from {model_path}")
            root.destroy()
            return
        
        # Variables
        self.image_path = None
        self.processed_image = None
        
        # Create GUI
        self.create_widgets()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Frame for the title
        title_frame = tk.Frame(self.root)
        title_frame.pack(pady=10)
        
        # Title
        title_label = tk.Label(
            title_frame, 
            text="Traffic Sign Recognition", 
            font=("Arial", 20, "bold")
        )
        title_label.pack()
        
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left frame for image
        self.image_frame = tk.Frame(main_frame, width=400, height=400, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        
        # Image label
        self.image_label = tk.Label(self.image_frame, text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right frame for buttons and prediction
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)
        
        # Button to upload image
        upload_button = tk.Button(
            control_frame, 
            text="Upload Image", 
            command=self.upload_image,
            font=("Arial", 12),
            height=2,
            bg="#4CAF50",
            fg="white"
        )
        upload_button.pack(fill=tk.X, pady=10)
        
        # Button to predict
        predict_button = tk.Button(
            control_frame, 
            text="Predict Sign", 
            command=self.predict,
            font=("Arial", 12),
            height=2,
            bg="#2196F3",
            fg="white"
        )
        predict_button.pack(fill=tk.X, pady=10)
        
        # Frame for prediction results
        self.result_frame = tk.Frame(control_frame, bd=2, relief=tk.GROOVE)
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Label for prediction title
        prediction_title = tk.Label(
            self.result_frame, 
            text="Prediction Result", 
            font=("Arial", 14, "bold")
        )
        prediction_title.pack(pady=10)
        
        # Label for sign name
        self.sign_label = tk.Label(
            self.result_frame, 
            text="", 
            font=("Arial", 12),
            wraplength=250
        )
        self.sign_label.pack(pady=5)
        
        # Label for confidence
        self.confidence_label = tk.Label(
            self.result_frame, 
            text="", 
            font=("Arial", 12)
        )
        self.confidence_label.pack(pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """Upload and display an image"""
        # Open file dialog to select an image
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm"),
                ("All files", "*.*")
            ]
        )
        
        if not self.image_path:
            return
        
        try:
            # Display the selected image
            self.display_image(self.image_path)
            self.status_bar.config(text=f"Loaded image: {os.path.basename(self.image_path)}")
            
            # Clear previous predictions
            self.sign_label.config(text="")
            self.confidence_label.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self, image_path):
        """Display the image on the GUI"""
        # Load and resize the image for display
        img = Image.open(image_path)
        img = img.resize((360, 360), Image.LANCZOS)
        
        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(img)
        
        # Update the image label
        self.image_label.config(image=photo)
        self.image_label.image = photo  # Keep a reference
    
    def predict(self):
        """Predict the traffic sign class"""
        if self.image_path is None:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        try:
            # Update status
            self.status_bar.config(text="Predicting...")
            self.root.update()
            
            # Preprocess the image
            img = preprocess_image(self.image_path)
            if img is None:
                messagebox.showerror("Error", "Failed to process the image")
                self.status_bar.config(text="Ready")
                return
            
            # Make a prediction
            predictions = self.model.predict(img)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx]
            
            # Get the class name
            class_name = SIGN_NAMES.get(class_idx, f"Unknown (Class {class_idx})")
            
            # Update the result labels
            self.sign_label.config(text=f"Sign: {class_name}")
            self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
            
            # Update status
            self.status_bar.config(text="Prediction complete")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
            self.status_bar.config(text="Ready")


def main():
    """
    Launch the Traffic Sign Recognition App.
    
    Usage: python traffic_sign_app.py model.h5
    """
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python traffic_sign_app.py model.h5")
    
    model_path = sys.argv[1]
    
    # Check if model file exists
    if not os.path.isfile(model_path):
        sys.exit(f"Error: Model file '{model_path}' does not exist")
    
    # Create the Tkinter window
    root = tk.Tk()
    app = TrafficSignApp(root, model_path)
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()