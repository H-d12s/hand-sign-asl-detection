# 🧠 Real-Time Sign Language Detection (ASL)

A deep learning-based system that detects American Sign Language (A–Z) using a webcam in real time.

---

## 🚀 Features

* 🔤 Classifies ASL alphabets (A–Z)
* 🎥 Real-time webcam prediction
* ✋ Hand detection using MediaPipe
* 🧠 CNN built from scratch using PyTorch
* ⚡ GPU acceleration (if available)

---

## 🧠 Tech Stack

* **PyTorch** – Model training & inference
* **OpenCV** – Webcam handling
* **MediaPipe** – Hand tracking
* **Python** – Core implementation

---


## ⚙️ Installation

```bash
git clone https://github.com/your-username/hand-sign-asl-detection.git
cd hand-sign-asl-detection
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### 🔹 Train Model

```bash
cd src
python train.py
```

### 🔹 Run Real-Time Detection

```bash
python app.py
```

---

## 📸 How It Works

```
Webcam → MediaPipe Hand Detection → Crop Hand → CNN → Prediction (A–Z)
```

---

## ⚠️ Limitations

* Sensitive to lighting conditions
* Accuracy depends on hand positioning
* Model trained on static dataset (not real-world variations)

---

## 🔥 Future Improvements

* Improve accuracy with augmented dataset
* Add prediction smoothing
* Support full word/sentence recognition
* Deploy as web/mobile app

---

## 👤 Author

Built as a deep learning + computer vision project to understand end-to-end model deployment.

---


