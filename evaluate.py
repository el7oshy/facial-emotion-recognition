import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset import ImageEmotionDataset
from model import CNNEmotion
from sklearn.metrics import confusion_matrix, classification_report

# Paths
base_path = "data"  # folder containing train/ and test/ (FER-2013)
test_dir = os.path.join(base_path, "test")
classes = sorted(os.listdir(test_dir))

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CNNEmotion(num_classes=len(classes))
model.load_state_dict(torch.load("models/cnn_model.pt", map_location=device))
model.to(device)
model.eval()

def evaluate_on_test_set():
    """Evaluate model on test dataset"""
    print("\n" + "="*50)
    print("Evaluating on Test Dataset...")
    print("="*50)
    
    # Dataset & Loader
    test_dataset = ImageEmotionDataset(test_dir, classes)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    all_preds = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Detailed Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

def preprocess_face(face_img):
    """Preprocess face image for model input"""
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    
    # Convert to tensor
    tensor_img = torch.tensor(resized / 255.0, dtype=torch.float32)
    tensor_img = (tensor_img - 0.5) / 0.5  # Normalize to [-1, 1]
    tensor_img = tensor_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    return tensor_img.to(device)

def predict_emotion(face_img):
    """Predict emotion from face image"""
    with torch.no_grad():
        tensor_img = preprocess_face(face_img)
        outputs = model(tensor_img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()

def real_time_emotion_detection(camera_index=0):
    """Real-time emotion detection using webcam"""
    print("\n" + "="*50)
    print("Starting Real-time Emotion Detection")
    print("="*50)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("="*50)
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        print("Trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No camera found!")
            return
    
    print("Camera initialized successfully")
    
    # Emotion colors (BGR format)
    emotion_colors = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 128, 0),    # Green
        'fear': (255, 0, 255),     # Magenta
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'surprise': (255, 165, 0), # Orange
        'neutral': (128, 128, 128) # Gray
    }
    
    # Get default color for unknown emotions
    default_color = (200, 200, 200)
    
    frame_count = 0
    emotion_history = []
    max_history = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Mirror the frame for more natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Predict emotion every 5 frames to reduce processing
                if frame_count % 5 == 0:
                    emotion_idx, confidence = predict_emotion(face_roi)
                    emotion = classes[emotion_idx]
                    emotion_history.append(emotion)
                    if len(emotion_history) > max_history:
                        emotion_history.pop(0)
                
                # Get color for the current emotion
                current_emotion = emotion_history[-1] if emotion_history else "neutral"
                color = emotion_colors.get(current_emotion, default_color)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion and confidence
                label = f"{current_emotion}"
                cv2.putText(frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display most common emotion in history
                if emotion_history:
                    from collections import Counter
                    most_common = Counter(emotion_history).most_common(1)[0][0]
                    cv2.putText(frame, f"Avg: {most_common}", (x, y+h+20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display emotion history
        history_text = "History: " + " ".join(emotion_history[-max_history:])
        cv2.putText(frame, history_text, (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Real-time Emotion Detection', frame)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera session ended.")

def single_image_emotion_detection(image_path):
    """Detect emotion in a single image"""
    print(f"\nAnalyzing image: {image_path}")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image")
        return
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    print(f"Found {len(faces)} face(s)")
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = img[y:y+h, x:x+w]
        
        if face_roi.size > 0:
            emotion_idx, confidence = predict_emotion(face_roi)
            emotion = classes[emotion_idx]
            
            print(f"Face {i+1}: {emotion} (confidence: {confidence*100:.1f}%)")
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion}: {confidence*100:.1f}%", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow('Emotion Detection Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Evaluate on test set
    evaluate_on_test_set()
    
    # Ask user what they want to do
    print("\n" + "="*50)
    print("What would you like to do?")
    print("1. Real-time camera emotion detection")
    print("2. Analyze emotion in an image file")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        # Try different camera indices
        camera_idx = 0
        camera_input = input(f"Enter camera index (default {camera_idx}): ").strip()
        if camera_input:
            camera_idx = int(camera_input)
        real_time_emotion_detection(camera_idx)
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            single_image_emotion_detection(image_path)
        else:
            print(f"Error: Image not found at {image_path}")
    else:
        print("Exiting...")