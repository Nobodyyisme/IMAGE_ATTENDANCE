# 👨‍🏫 Image-Based Face Attendance System

A smart face recognition-based **automated attendance system** that detects faces in images or videos, extracts embeddings, and identifies individuals using a trained classifier.

---

## 📌 Features

- 📷 **Face Detection** (via RetinaFace)
- 🧠 **Face Embedding Extraction** (via ArcFace ONNX model)
- ✅ **Classification** using SVM
- 🗃️ **Attendance marking** from video or image inputs
- 🧪 **Evaluation** via confusion matrix and metrics

---

## 🗂️ Project Structure

```
WITH_EMBEDDING/
├── dataset/                   # Organized training dataset (aligned faces per person)
├── input/                     # Input images or video
├── models/
│   ├── model.onnx             # ArcFace ONNX model
│   ├── svm_classifier.joblib  # Trained classifier
│   ├── label_encoder.joblib   # Label encoder for name ↔ index mapping
├── raw_images/                # Raw cropped/processed face images
├── raw_videos/                # Input videos (e.g., from CCTV or webcam)
├── test/
│   ├── dataset/               # Test dataset
│   ├── raw_images/            # Raw test face images
│   ├── trained_xy/            # Saved embeddings and labels
│   ├── X_embeddings.npy       # Test features
│   ├── y_labels.npy           # Test labels
├── output.mp4                 # Output video with annotations
├── confusion_matrix.png       # Evaluation plot
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

Organize your face images in this format:

```
dataset/
├── Person1/
│   ├── img1.jpg
│   ├── img2.jpg
├── Person2/
```

### 3. Train the model

```python
# Extract embeddings and train classifier
extract_and_save_embeddings(data_dir="./dataset", save_dir="./trained_xy")
train_svm_classifier(X_path="./trained_xy/X_embeddings.npy", y_path="./trained_xy/y_labels.npy")
```

### 4. Run prediction on a video

```python
predict_from_video("raw_videos/M.mp4", output_path="output.mp4")
```

---

## 📊 Evaluation

- Classification accuracy
- Confusion matrix (saved as `confusion_matrix.png`)
- Real-time video annotation

---

## 📦 Dependencies

- opencv-python
- insightface
- numpy
- joblib
- scikit-learn
- onnxruntime

---
