import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def preprocess_for_cnn(image, size=(64, 64)):
    """
    Standardizes an image for the CNN: converts to grayscale and resizes.
    """
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to the target size
    resized_image = cv2.resize(gray_image, size)
    return resized_image

def preprocess_for_knn(image, size=(256, 256)):
    """
    Segments the hand from the background to create clean input for MediaPipe.
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a range for skin color in HSV
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask to identify skin color
    mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((4, 4), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black background
    segmented_image = np.zeros_like(image)
    
    if contours:
        # Find the largest contour by area (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        hand_mask = np.zeros_like(mask)
        cv2.drawContours(hand_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
        
        # Apply the final mask to the original image
        segmented_image = cv2.bitwise_and(image, image, mask=hand_mask)
        
    # Resize the final segmented image
    return cv2.resize(segmented_image, size)

# --- Main Script Logic ---
if __name__ == "__main__":
    # IMPORTANT: Set this to the folder containing your captured images
    input_dir = 'data'
    
    # Define output directories
    output_dir_cnn = 'processed_for_cnn'
    output_dir_knn = 'processed_for_knn'

    # Create output directories if they don't exist
    os.makedirs(output_dir_cnn, exist_ok=True)
    os.makedirs(output_dir_knn, exist_ok=True)
    
    data_dir = Path(input_dir)
    if not data_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        exit()
        
    label_folders = [d for d in data_dir.iterdir() if d.is_dir()]

    for label_folder in tqdm(label_folders, desc="Processing Letters"):
        label = label_folder.name
        
        # Create corresponding subfolders in the output directories
        os.makedirs(Path(output_dir_cnn) / label, exist_ok=True)
        os.makedirs(Path(output_dir_knn) / label, exist_ok=True)

        for img_path in label_folder.glob('*.jpg'):
            # Load the original image
            original_image = cv2.imread(str(img_path))
            if original_image is not None:
                # Process for CNN
                cnn_image = preprocess_for_cnn(original_image)
                cnn_save_path = Path(output_dir_cnn) / label / img_path.name
                cv2.imwrite(str(cnn_save_path), cnn_image)
                
                # Process for k-NN/MediaPipe
                knn_image = preprocess_for_knn(original_image)
                knn_save_path = Path(output_dir_knn) / label / img_path.name
                cv2.imwrite(str(knn_save_path), knn_image)

    print("\nPreprocessing complete!")
    print(f"CNN-ready images saved in '{output_dir_cnn}'")
    print(f"k-NN/MediaPipe-ready images saved in '{output_dir_knn}'")