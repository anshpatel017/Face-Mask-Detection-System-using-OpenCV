# 😷 Face Mask Detection System using OpenCV

A real-time face mask detection system that identifies whether a person is
**wearing a face mask** or **not** through a webcam, video, or image. It
combines **OpenCV** for face detection with a **deep learning classifier**
(MobileNetV2) trained to distinguish masked from unmasked faces.

---

## 📋 Overview

The system works in two stages:

1. **Face Detection** — OpenCV locates all faces in the frame using a
   pre-trained face detector (Caffe-based SSD / Haar Cascade).
2. **Mask Classification** — each detected face is passed to a CNN model that
   predicts `Mask` or `No Mask` along with a confidence score.

Results are drawn live on the video feed: a **green box** for faces with a mask
and a **red box** for faces without one, each labelled with the prediction
confidence.

---

## ✨ Features

- **Real-time detection** through a webcam feed
- Works on **images, recorded videos, and live camera**
- Detects **multiple faces** simultaneously in a single frame
- **Color-coded bounding boxes** (green = mask, red = no mask) with confidence %
- Lightweight **MobileNetV2** backbone for fast inference
- Includes both a **training pipeline** and a **detection pipeline**

---

## 🧠 Model Architecture

The classifier is built on **MobileNetV2** using transfer learning:

```
Input image (224 × 224 × 3)
        │
   MobileNetV2 (pre-trained on ImageNet, base frozen)
        │
   AveragePooling2D (7 × 7)
        │
   Flatten
        │
   Dense (128 units) + ReLU
        │
   Dropout (0.5)
        │
   Dense (2 units) + Softmax  →  [Mask, No Mask]
```

### Training Configuration

| Setting | Value |
|---|---|
| Base model | MobileNetV2 (ImageNet weights) |
| Input size | 224 × 224 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| Loss | Binary cross-entropy |
| Data augmentation | Rotation, zoom, shift, flip |

---

## 📂 Project Structure

```
Face Mask Detection System using OpenCV/
├── dataset/
│   ├── with_mask/            # Images of people wearing masks
│   └── without_mask/         # Images of people without masks
├── face_detector/
│   ├── deploy.prototxt        # Face detector architecture
│   └── res10_300x300_ssd.caffemodel  # Pre-trained face detector weights
├── train_mask_detector.py     # Trains the mask classifier
├── detect_mask_video.py       # Real-time detection from webcam
├── detect_mask_image.py       # Detection on a single image
├── mask_detector.model        # Saved trained model
├── plot.png                   # Training accuracy/loss curves
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset

The model is trained on a dataset of face images split into two classes:

- **`with_mask`** — faces correctly wearing a face mask
- **`without_mask`** — faces without a mask

The dataset is augmented during training (rotations, zoom, shifts, and
horizontal flips) to improve generalization and reduce overfitting.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** — face detection and video/image processing
- **TensorFlow / Keras** — deep learning model (MobileNetV2)
- **NumPy** — array operations
- **imutils** — image/video utility helpers
- **Matplotlib** — plotting training curves
- **scikit-learn** — data splitting and evaluation metrics

---

## ⚙️ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Face Mask Detection System using OpenCV"

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```
tensorflow>=2.5.0
keras
opencv-python
imutils
numpy
matplotlib
scikit-learn
```

---

## ▶️ Usage

### 1. Train the model
```bash
python train_mask_detector.py
```
This trains the classifier on the dataset, saves `mask_detector.model`, and
generates `plot.png` showing the accuracy/loss curves.

### 2. Real-time detection (webcam)
```bash
python detect_mask_video.py
```
Opens the webcam and labels each detected face as **Mask** or **No Mask** in
real time. Press **`q`** to quit.

### 3. Detection on a single image
```bash
python detect_mask_image.py --image examples/test.jpg
```

---

## 📊 Results

The model achieves high accuracy on the validation set and runs in real time
on a standard CPU. Detection output shows:

- 🟩 **Green box** → face with mask
- 🟥 **Red box** → face without mask
- Confidence percentage on each label

---

## 🚀 Future Improvements

- Add a third class for **improperly worn masks** (mask below nose/chin)
- Deploy as a **web app** (Flask / Streamlit) or mobile application
- Integrate with **CCTV / IP cameras** for crowd monitoring
- Trigger **alerts** when a person without a mask is detected

---

## 📄 License

This project is intended for **educational and research purposes**.
