import torch.nn as nn
import torch
import cv2
import numpy as np

class CNNEmotion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 48 → 24

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24 → 12

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12 → 6
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    def predict_from_camera_frame(self, face_roi, device='cpu'):
        """
        Predict emotion from a camera frame face ROI
        
        Args:
            face_roi: BGR image from camera (numpy array)
            device: 'cpu' or 'cuda'
        
        Returns:
            predicted_class (int), confidence (float), all_probabilities (numpy array)
        """
        # Preprocess the face ROI
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        
        # Convert to tensor
        tensor_img = torch.tensor(resized / 255.0, dtype=torch.float32)
        tensor_img = (tensor_img - 0.5) / 0.5  # Normalize to [-1, 1]
        tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        # Move to device
        tensor_img = tensor_img.to(device)
        self.to(device)
        
        # Predict
        with torch.no_grad():
            self.eval()
            outputs = self(tensor_img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def get_top_emotions(self, face_roi, class_names, device='cpu', top_k=3):
        """
        Get top K predicted emotions with confidence scores
        
        Args:
            face_roi: BGR image from camera
            class_names: List of emotion class names
            device: 'cpu' or 'cuda'
            top_k: Number of top predictions to return
        
        Returns:
            List of tuples (emotion_name, confidence)
        """
        predicted, confidence, all_probs = self.predict_from_camera_frame(face_roi, device)
        
        # Get indices of top K probabilities
        top_indices = np.argsort(all_probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            emotion_name = class_names[idx]
            prob = all_probs[idx]
            results.append((emotion_name, prob))
        
        return results

# Camera utility function
def quick_camera_test(model_path=None, class_names=None, camera_index=0):
    """
    Quick test of emotion detection on camera feed
    
    Args:
        model_path: Path to saved model (optional)
        class_names: List of emotion class names
        camera_index: Camera device index
    """
    import cv2
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Load model if path provided
    model = None
    device = 'cpu'
    if model_path:
        model = CNNEmotion(num_classes=len(class_names) if class_names else 7)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # If model is loaded, predict emotion
            if model and class_names:
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    try:
                        results = model.get_top_emotions(face_roi, class_names, device, top_k=1)
                        emotion, confidence = results[0]
                        
                        # Display prediction
                        label = f"{emotion}: {confidence*100:.1f}%"
                        cv2.putText(frame, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Prediction error: {e}")
        
        cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()