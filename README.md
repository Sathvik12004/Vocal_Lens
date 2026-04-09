#  Vocal Lens — Smart Glasses for the Visually Impaired

> A real-time, voice-interactive assistive vision system powered by YOLOv8, running entirely on a CPU-only machine.

---

## 📌 Overview

**Vocal Lens** is a smart glasses assistant designed to help visually impaired users navigate their environment safely. It uses a webcam feed to detect objects and obstacles in real time, communicates via text-to-speech, and responds to natural voice commands — no wake word required.

This project was built as part of assistive robotics research and serves as the software backbone for a wearable smart glasses prototype.

---

## ✨ Features

- 🔍 **Real-time object detection** using YOLOv8n (runs at ~2s intervals for efficiency)
- 🗣️ **Voice command interface** with fuzzy keyword matching (no rigid phrases needed)
- 📢 **Text-to-speech feedback** via `pyttsx3` (fully offline)
- 🚧 **Danger zone alerts** — presence-based obstacle detection in the user's forward path
- 📖 **OCR sign reading** via `pytesseract` (optional)
- 🧠 **State machine architecture** — clean transitions between IDLE, OVERVIEW, WALKING, and END states
- 💻 **CPU-only** — no GPU required

---

## 🗣️ Voice Commands

| Command | What it does |
|---|---|
| `"introduce"` | Vocal Lens introduces itself |
| `"start"` / `"trip"` | Scans surroundings and gives an overview |
| `"yes"` / `"ready"` | Begins live walking guidance |
| `"what do you see"` | Describes current surroundings |
| `"how many people"` | Counts people in frame |
| `"is the path clear"` | Checks if the forward path is obstacle-free |
| `"read signs"` | Reads visible text via OCR |
| `"end"` / `"stop trip"` | Shuts down gracefully |

---

## 🏗️ System Architecture

```
Webcam Feed
    │
    ▼
YOLOv8n Detection (every 2s)
    │
    ├── Danger Zone Analysis (center 33% of frame)
    │       └── Presence-based alerts → VoiceEngine
    │
    └── Detection Cache (thread-safe)
            │
            ▼
      Voice Commands (Microphone Listener)
            │
            ▼
      State Machine Router
      (IDLE → OVERVIEW → WALKING → END)
            │
            ▼
      pyttsx3 TTS Output
```

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/vocal-lens.git
cd vocal-lens
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Install Tesseract for OCR sign reading
- **Windows**: Download from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt install tesseract-ocr`
- **macOS**: `brew install tesseract`

> The YOLOv8n model (`yolov8n.pt`) will be **automatically downloaded** on first run via the `ultralytics` package.

---

## 🚀 Usage

```bash
python main.py
```

Vocal Lens will greet you, start the camera feed, and wait for your voice commands.

Press **`Q`** or say **`"end"`** to shut down.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `ultralytics` | YOLOv8 object detection |
| `opencv-python` | Camera capture & HUD rendering |
| `pyttsx3` | Offline text-to-speech |
| `SpeechRecognition` | Microphone voice input |
| `pytesseract` | OCR for sign reading (optional) |

See `requirements.txt` for pinned versions.

---

## 📁 Project Structure

```
vocal-lens/
├── main.py              # Main application
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore
```

---

## 🔭 Future Work (Vocal Lens v2)

- [ ] Replace `pyttsx3` with a neural TTS voice (e.g., Coqui TTS)
- [ ] Add depth estimation for distance-aware alerts
- [ ] Integrate GPS for outdoor navigation
- [ ] Port to Raspberry Pi / Jetson Nano for true wearable deployment
- [ ] Custom fine-tuned YOLO model for indoor-specific objects

---

## 👤 Author

**G. Sathvik**
B.Tech ECE — Vignana Bharathi Institute of Technology (JNTUH), Hyderabad
GitHub: [github.com/Sathvik12004](https://github.com/Sathvik12004)

---

## 📄 License

This project is licensed under the MIT License.
