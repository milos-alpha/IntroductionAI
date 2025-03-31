import os
import shutil
import argparse
import urllib.request
import cv2
import numpy as np
import random

def download_sample_images(target_dir="images", num_images=10):
    """
    Download sample traffic sign images from the web and save them to target directory
    with appropriate naming convention.
    
    Args:
        target_dir (str): Directory to save downloaded images
        num_images (int): Number of images to download
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Sample image URLs and their corresponding categories
    # These are just examples - update with actual image URLs if needed
    print("Note: This function requires internet access to download sample images.")
    print("If you prefer to use your own images, press Ctrl+C and use the collect_from_directory function instead.")
    
    # This would be much better with actual image URLs, but for now we'll create
    # synthetic images for demonstration purposes
    for i in range(num_images):
        category = random.randint(0, 42)
        
        # Create a simple synthetic image (this is just for demonstration)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Draw a circle for the sign
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img, (100, 100), 80, color, -1)
        
        # Draw a border
        cv2.circle(img, (100, 100), 80, (255, 255, 255), 5)
        
        # Add text with the category number
        cv2.putText(img, str(category), (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (255, 255, 255), 4, cv2.LINE_AA)
        
        # Save the image
        filename = f"{category}_image_{i+1}.jpg"
        filepath = os.path.join(target_dir, filename)
        cv2.imwrite(filepath, img)
        
        print(f"Created sample image: {filename}")
    
    print(f"Successfully created {num_images} sample images in {target_dir}")
    print("Note: These are synthetic images for demonstration. Replace them with real traffic sign images for actual use.")


def collect_from_directory(source_dir, target_dir="images"):
    """
    Collect images from a directory and organize them for testing.
    
    Args:
        source_dir (str): Directory containing source images
        target_dir (str): Directory to save organized images
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # List all files in source directory
    files = os.listdir(source_dir)
    
    # Filter image files
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.ppm'))]
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} image files.")
    print("For each image, enter the category number (0-42) of the traffic sign.")
    print("Enter 'q' to quit the process.")
    
    count = 0
    for i, file in enumerate(image_files):
        source_path = os.path.join(source_dir, file)
        
        # Display the image if possible
        try:
            img = cv2.imread(source_path)
            if img is not None:
                # Resize if too large for display
                h, w = img.shape[:2]
                if h > 600 or w > 800:
                    scale = min(600/h, 800/w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                
                cv2.imshow("Image", img)
                cv2.waitKey(100)  # Short delay to ensure window updates
        except Exception as e:
            print(f"Could not display image {file}: {e}")
        
        # Ask the user for the category
        category = input(f"Enter category for image {file} (0-42, or 'q' to quit): ")
        
        if category.lower() == 'q':
            break
        
        try:
            category_num = int(category)
            if not (0 <= category_num < 43):
                print("Category must be between 0 and 42. Skipping this image.")
                continue
                
            # Create a new filename with category prefix
            ext = os.path.splitext(file)[1]
            new_filename = f"{category_num}_image_{i+1}{ext}"
            target_path = os.path.join(target_dir, new_filename)
            
            # Copy the file to the target directory with the new name
            shutil.copy(source_path, target_path)
            print(f"Copied {file} to {new_filename}")
            count += 1
            
        except ValueError:
            print(f"Invalid category '{category}'. Please enter a number between 0 and 42.")
    
    if count > 0:
        print(f"Successfully collected {count} test images in {target_dir}")
    else:
        print("No images were collected.")
    
    # Close the display window if it was opened
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Collect and organize test images for traffic sign recognition")
    parser.add_argument("--source", help="Directory containing source images", default=None)
    parser.add_argument("--target", help="Target directory to save organized images", default="images")
    parser.add_argument("--num", type=int, help="Number of sample images to generate", default=10)
    
    args = parser.parse_args()
    
    if args.source:
        collect_from_directory(args.source, args.target)
    else:
        download_sample_images(args.target, args.num)


if __name__ == "__main__":
    main()