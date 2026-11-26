import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageEmotionDataset(Dataset):
    def __init__(self, root_dir, classes, augment=False):
        self.root_dir = root_dir
        self.classes = classes
        self.augment = augment
        self.data = []

        # Augmentation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        for label, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.data.append((os.path.join(cls_path, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))

        if self.augment:
            img = self.transform(img)
        else:
            img = torch.tensor(img / 255.0, dtype=torch.float32).unsqueeze(0)
            img = (img - 0.5) / 0.5   # Normalize

        return img, torch.tensor(label, dtype=torch.long)

# Camera utility functions
def capture_from_camera(camera_index=0, save_dir=None):
    """Capture images from camera for data collection"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("Press 'c' to capture, 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Display instructions
        cv2.putText(frame, "Press 'c' to capture", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Captured: {frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Capture Images', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if save_dir:
                # Save the captured frame
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {frame_count} images")

def test_camera_feed(camera_index=0):
    """Test camera feed with face detection preview"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return
    
    print("Testing camera feed with face detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera Test with Face Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()