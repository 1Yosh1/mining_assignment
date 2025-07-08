Overview
This project centers on the development of a computer vision system designed to recognize and classify individual letters of the Amharic sign language alphabet. The work addresses several common machine learning challenges, including data scarcity, file system compatibility issues, and memory constraints.

The project began with a k-Nearest Neighbors (k-NN) classifier leveraging MediaPipe landmark features. However, due to brittleness and limited performance, the methodology pivoted to a more robust, end-to-end Convolutional Neural Network (CNN) approach. After assembling a custom dataset of sign language images, the final CNN model was trained and achieved high accuracy on a test set of previously unseen images. The project culminates in a real-time application capable of capturing webcam images and predicting the corresponding Amharic sign language letter.

Features
Real-time webcam capture and letter prediction
Custom dataset creation for Amharic sign language
Transition from k-NN with MediaPipe to CNN for improved robustness
High-accuracy classification of unseen test data
Dependencies
The following Python packages are required (see requirements.txt):

opencv-python
mediapipe
numpy
scikit-learn
matplotlib
seaborn
Install all dependencies with:

bash
pip install -r requirements.txt
Getting Started
Clone the repository:

bash
git clone https://github.com/1Yosh1/mining_assignment.git
cd mining_assignment
Install dependencies:

bash
pip install -r requirements.txt
Run the application:

The main script (e.g., main.py) will start the real-time webcam classifier.
Ensure your webcam is connected.
bash
python main.py
Project Structure
main.py – Main application script
requirements.txt – List of Python dependencies
(Add details about other important files or folders, e.g., dataset, model files, etc.)
Usage
Launch the application and follow on-screen instructions to capture and classify Amharic sign language letters in real time.
For training or retraining the model, ensure your image dataset is prepared as described in the source code.
