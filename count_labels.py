import cv2
import os
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
from collections import Counter

def extract_landmarks(image):
    """Reads an image and returns True if a hand is found, False otherwise."""
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        return results.multi_hand_landmarks is not None

if __name__ == "__main__":
    base_dir = 'processed_for_knn'
    data_dir = Path(base_dir)

    if not data_dir.is_dir():
        print(f"Error: Directory not found at '{data_dir}'.")
        exit()

    labels = []
    all_image_paths = list(data_dir.glob('*/*.jpg')) + list(data_dir.glob('*/*.png'))

    print(f"Checking {len(all_image_paths)} images...")
    
    for img_path in tqdm(all_image_paths, desc="Checking Images"):
        label = img_path.parent.name
        image = cv2.imread(str(img_path))
        if image is not None:
            # Check if MediaPipe can find a hand
            if extract_landmarks(image):
                labels.append(label)

    print("\n--- Feature Extraction Success Count per Letter ---")
    label_counts = Counter(labels)
    
    found_problem = False
    for label, count in sorted(label_counts.items()):
        print(f"  - Letter '{label}': {count} usable samples")
        if count < 2:
            print(f"    !!!! PROBLEM: This class has fewer than 2 samples. !!!!")
            found_problem = True
            
    if not found_problem:
        print("\n✅ All classes have enough samples.")