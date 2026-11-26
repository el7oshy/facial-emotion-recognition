import os
import torch
import math
import cv2
import numpy as np
from torch.utils.data import DataLoader
from collections import Counter

from dataset import ImageEmotionDataset
from model import CNNEmotion

# ---------------- Paths ----------------
train_dir =r"D:\MNU\Semster 5\Computer Vision\Project\train"
test_dir  = r"D:\MNU\Semster 5\Computer Vision\Project\test"

classes = sorted(os.listdir(train_dir))

# ---------------- Datasets ----------------
train_dataset = ImageEmotionDataset(train_dir, classes, augment=True)
test_dataset  = ImageEmotionDataset(test_dir, classes, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Model ----------------
model = CNNEmotion(num_classes=len(classes)).to(device)

# ---------------- Class Weights ----------------
labels = [label for _, label in train_dataset.data]
counts = Counter(labels)
total = sum(counts.values())

class_weights = [
    math.sqrt(total / counts[i]) for i in range(len(counts))
]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ---------------- Loss & Optimizer ----------------
criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# ---------------- Training ----------------
epochs = 25
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    scheduler.step(epoch_loss)

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

# ---------------- Save Model ----------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/cnn_model.pt")

print("Model saved successfully.")

# ---------------- Camera Test Function ----------------
def test_camera_access():
    """Test if camera is accessible and show preview"""
    print("\n" + "="*50)
    print("Testing Camera Access...")
    print("="*50)
    
    # Try different camera indices
    for camera_idx in range(3):  # Try first 3 cameras
        cap = cv2.VideoCapture(camera_idx)
        
        if cap.isOpened():
            print(f"✓ Camera {camera_idx} is accessible")
            
            # Read a test frame
            ret, frame = cap.read()
            if ret:
                print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
                
                # Show preview
                cv2.putText(frame, f"Camera {camera_idx}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(f'Camera {camera_idx} Preview', frame)
                cv2.waitKey(2000)  # Show for 2 seconds
                cv2.destroyAllWindows()
            else:
                print(f"  Failed to capture frame from camera {camera_idx}")
            
            cap.release()
        else:
            print(f"✗ Camera {camera_idx} is not accessible")
    
    print("="*50)

# Run camera test after training
test_camera_access()