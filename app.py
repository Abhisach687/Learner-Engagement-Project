import streamlit as st
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Load trained model (Placeholder - Replace with actual model loading code)
def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.fc = torch.nn.Linear(2048, 3)  # Assuming 3 classes: bored, attentive, confused
    model.load_state_dict(torch.load("models/model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess frame
def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(frame).unsqueeze(0)

# Perform inference
def predict_emotion(model, frame):
    frame = preprocess_frame(frame)
    with torch.no_grad():
        output = model(frame)
        prediction = torch.argmax(output, dim=1).item()
    return ["Bored", "Attentive", "Confused"][prediction]

st.title("Real-Time Learner Engagement Detection")

if st.button("Start Webcam"):
    cap = cv2.VideoCapture(0)
    model = load_model()

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prediction = predict_emotion(model, frame_rgb)

        cv2.putText(frame, f"Engagement: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        stframe.image(frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()