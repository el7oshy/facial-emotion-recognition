# Facial Emotion Recognition System

Real-time facial emotion recognition with a custom CNN built in PyTorch. It trains on a folder-based image dataset and runs live on a webcam, using OpenCV to detect faces.

**Course project:** Computer Vision (Solo) | **Stack:** Python, PyTorch, OpenCV, NumPy, scikit-learn

## Results
- Dataset: **FER-2013** (Kaggle), 7 emotion classes. Test accuracy: **58.37%** (7,178 test images; macro F1 0.53). Best classes: happy (F1 0.81) and surprise (0.71); hardest: fear (0.30).
- Evaluation prints the confusion matrix and a per-class precision/recall/F1 report.

## How it works
1. **Face detection:** OpenCV Haar Cascade finds faces in each webcam frame.
2. **Preprocessing:** the face crop is converted to grayscale, resized to 48x48 and normalized to [-1, 1]. Training and inference use the same normalization.
3. **Model (`model.py`):** a 3-block CNN (Conv, BatchNorm, ReLU, MaxPool with 32, 64 and 128 filters), then Flatten, Linear(256), Dropout(0.5) and a Linear output layer over the emotion classes.
4. **Prediction:** softmax probabilities give the top emotion and a confidence score. Real-time mode predicts every 5th frame and shows a short history of recent predictions.

## Training choices (`train.py`)
- **Class weighting:** square root of the inverse class frequency, to handle imbalanced emotion classes.
- **Label smoothing (0.1):** reduces overconfident predictions.
- **Augmentation:** random rotation of 10 degrees. No horizontal flip.
- **Optimizer:** Adam (lr 3e-4) with `ReduceLROnPlateau`, batch size 64, 25 epochs.

## Project structure
| File | Purpose |
|---|---|
| `dataset.py` | `ImageEmotionDataset` loader and preprocessing, camera capture helpers |
| `model.py` | `CNNEmotion` network and camera prediction helpers |
| `train.py` | Training loop and model saving to `models/cnn_model.pt` |
| `evaluate.py` | Test-set evaluation, webcam demo, single-image prediction |
| `Documentation.pdf` | Full design documentation |

## Run it
```bash
pip install torch torchvision opencv-python numpy scikit-learn
# Dataset layout: train/<emotion>/*.jpg and test/<emotion>/*.jpg
python train.py
python evaluate.py      # evaluate, then choose webcam or single-image mode
```
Set `train_dir` and `test_dir` at the top of `train.py` and `evaluate.py` to your dataset location.

## Possible improvements
- Replace Haar Cascade with a DNN-based face detector for non-frontal faces.
- Add validation-based early stopping and checkpoints.
- Export the trained model to ONNX for lighter deployment.
