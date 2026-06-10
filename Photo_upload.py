
# image_emotion_local.py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import json
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from model_architecture import EmotionCNN

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'best_emotion_model_weights.pth'
EMOTION_LABELS_PATH = 'emotion_labels.json'
IMG_SIZE = 48

# 1. Select Image File using Windows/Mac/Linux file dialog
root = tk.Tk()
root.withdraw() # Hide the main tkinter window
file_path = filedialog.askopenfilename(title="Select an image for emotion detection")

if not file_path:
    print("No file selected. Exiting.")
    exit()

# 2. Load Model and Labels
with open(EMOTION_LABELS_PATH, 'r') as f:
    emotion_labels = json.load(f)
num_classes = len(emotion_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()

# 3. Prepare Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Process Image
img_bgr = cv2.imread(file_path)
if img_bgr is None:
    print("Could not read the image.")
    exit()

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

for (x, y, w, h) in faces:
    # Crop face
    roi_gray = img_gray[y:y+h, x:x+w]
    pil_face = Image.fromarray(roi_gray)

    # Transform and Predict
    input_tensor = transform(pil_face).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prob, idx = torch.max(probabilities, 1)

    # Draw result on image
    label = f"{emotion_labels[idx.item()]}: {prob.item()*100:.1f}%"
    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_bgr, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 5. Show Results
cv2.imshow(f'Detected {len(faces)} face(s)', img_bgr)
print("Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()