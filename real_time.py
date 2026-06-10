
# updated realtime_emotion.py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image
from model_architecture import EmotionCNN

# --- Configuration ---
MODEL_WEIGHTS_PATH = 'best_emotion_model_weights.pth'
EMOTION_LABELS_PATH = 'emotion_labels.json'
IMG_SIZE = 48

# Load Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Load Model and Labels ---
with open(EMOTION_LABELS_PATH, 'r') as f:
    emotion_labels = json.load(f)
num_classes = len(emotion_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
model.eval()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect multiple faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the face ROI (Region of Interest)
        roi_gray = gray_frame[y:y+h, x:x+w]
        pil_image = Image.fromarray(roi_gray)

        # Preprocess and Predict
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob, idx = torch.max(probabilities, 1)

        label = f"{emotion_labels[idx.item()]}: {prob.item()*100:.1f}%"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Multi-Face Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()