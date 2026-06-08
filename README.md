# HandTalk 🤟

A real-time Hand Sign Language Detector that recognizes hand gestures and converts them into audio output — bridging communication between deaf and non-deaf individuals.

Built as a project for the subject **Operating Systems**.

---

## What it does

- Detects hand gestures in real-time using your webcam
- Recognizes individual **letters** and **words** from sign language
- Converts recognized signs into **audio output** using text-to-speech
- Enables seamless communication between deaf and non-deaf people

---

## How it works

Webcam Input
↓
Hand Detection (MediaPipe)
↓
Frame Preprocessing (OpenCV)
↓
3D CNN Model Prediction
↓
Recognized Letter / Word
↓
Audio Output (pyttsx3)

---

## Tech Stack

| Layer | Library |
|---|---|
| Hand Detection | MediaPipe |
| Image Processing | OpenCV, Pillow |
| Model Training | TensorFlow, Keras |
| Data Handling | NumPy, Pandas |
| Text-to-Speech | pyttsx3 |
| Metrics & Preprocessing | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Utilities | tqdm, joblib, h5py |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/LRien/CRAFTED-QUARTERS-.git
cd HandTalk
```

**2. Install all dependencies**
```bash
pip install numpy pandas matplotlib seaborn
pip install opencv-python opencv-contrib-python mediapipe Pillow
pip install tensorflow scikit-learn
pip install nltk pyttsx3 SpeechRecognition
pip install tqdm joblib h5py
```

---

## Requirements

- Python 3.8+
- Webcam
- Windows / macOS / Linux

---

## Project Structure
HandTalk/
├── data/               ← training gesture images/videos
├── model/              ← saved trained model
├── src/
│   ├── detector.py     ← hand detection and landmark extraction
│   ├── predictor.py    ← model prediction logic
│   ├── audio.py        ← text-to-speech output
│   └── main.py         ← main application entry point
├── notebooks/          ← training and testing notebooks
├── requirements.txt    ← all dependencies
└── README.md

---

## Usage

```bash
python src/main.py
```

Point your webcam at your hand and start signing — HandTalk will detect your gestures and speak them out loud in real-time.

---

## Subject

**Operating Systems** — This project demonstrates real-time process handling, hardware interfacing (webcam), and inter-process communication between the detection, prediction, and audio output layers.

---

## Developer

**Leonardo Enricho Quadra**  
GitHub: [@LRien](https://github.com/LRien)  
LinkedIn: [leonardo-enricho-quadra-04a921213](https://linkedin.com/in/leonardo-enricho-quadra-04a921213)
