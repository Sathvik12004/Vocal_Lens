"""
Smart Glasses - Darius Mk.1  (v5)
===================================
State-based Interactive Voice Assistant + Object Detection

STATES:
  IDLE      → Greets user. Waits for "introduce" or "start"
  INTRODUCE → Introduces itself. Returns to IDLE
  OVERVIEW  → Scans surroundings, gives overview, asks "ready to walk?"
  WALKING   → Live obstacle feedback every 2 seconds. Ends on "end"
  END       → Shuts down gracefully

Voice Commands (fuzzy keyword matching — no wake word needed):
  "introduce"          → Self introduction
  "start" / "trip"     → Begin overview + ask if ready
  "yes" / "ready"      → Start live walking feedback
  "what do you see"    → Describe surroundings (works during WALKING)
  "how many people"    → Count people (works during WALKING)
  "is the path clear"  → Check ahead (works during WALKING)
  "read signs"         → OCR sign reading (works during WALKING)
  "end" / "stop trip"  → End session and shut down
"""

import cv2
import pyttsx3
import threading
import time
import queue
from ultralytics import YOLO

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed. Run: pip install SpeechRecognition")

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# ─────────────────────────────────────────────
# States
# ─────────────────────────────────────────────
class State:
    IDLE         = "IDLE"
    INTRODUCING  = "INTRODUCING"
    OVERVIEW     = "OVERVIEW"
    WALKING      = "WALKING"
    END          = "END"


# ─────────────────────────────────────────────
# Voice Engine
# ─────────────────────────────────────────────
class VoiceEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 160)
        self.engine.setProperty("volume", 1.0)
        self._queue = queue.Queue()
        self._running = True
        self._speaking = False
        threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text: str, priority: bool = False):
        if priority:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        print(f"[DARIUS] {text}")
        self._queue.put(text)

    def speak_and_wait(self, text: str):
        """Speak and block until finished — used for greetings/questions."""
        done = threading.Event()
        def _task():
            self.engine.say(text)
            self.engine.runAndWait()
            done.set()
        print(f"[DARIUS] {text}")
        threading.Thread(target=_task, daemon=True).start()
        done.wait()

    def is_speaking(self):
        return not self._queue.empty()

    def _run(self):
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                self._speaking = True
                self.engine.say(text)
                self.engine.runAndWait()
                self._speaking = False
            except queue.Empty:
                continue

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
# Microphone Listener
# ─────────────────────────────────────────────
class Listener:
    def __init__(self, callback):
        self.callback = callback   # called with recognized text (str)
        self._running = False
        self._thread = None
        self._paused = False       # pause listening while Darius is speaking

    def start(self):
        if not SPEECH_AVAILABLE:
            print("[WARN] Voice commands unavailable.")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[INFO] Microphone listener started.")

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def _loop(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 3000
        recognizer.dynamic_energy_threshold = True
        mic = sr.Microphone()

        with mic as source:
            print("[INFO] Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[INFO] Microphone ready.")

        while self._running:
            if self._paused:
                time.sleep(0.2)
                continue
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=6)
                text = recognizer.recognize_google(audio).lower().strip()
                print(f"[MIC] Heard: '{text}'")
                self.callback(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[ERROR] SR error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"[ERROR] Listener: {e}")
                time.sleep(1)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────
# Spatial Helpers
# ─────────────────────────────────────────────
def get_position(cx, frame_width):
    t = frame_width / 3
    if cx < t:
        return "left"
    elif cx < 2 * t:
        return "center"
    return "right"

def is_in_danger_zone(cx, frame_width):
    return (frame_width * 0.33) < cx < (frame_width * 0.66)

def describe_detections(detections):
    """Build a natural language summary from detections list."""
    if not detections:
        return "I don't see anything notable."
    counts = {}
    for det in detections:
        counts[det["label"]] = counts.get(det["label"], 0) + 1
    parts = []
    for label, count in counts.items():
        if count == 1:
            article = "an" if label[0] in "aeiou" else "a"
            parts.append(f"{article} {label}")
        else:
            parts.append(f"{count} {label}s")
    if len(parts) == 1:
        return f"I can see {parts[0]}."
    elif len(parts) == 2:
        return f"I can see {parts[0]} and {parts[1]}."
    return "I can see " + ", ".join(parts[:-1]) + f", and {parts[-1]}."

def path_status(detections, frame_width):
    """Returns (is_clear, message) about the center path."""
    obstacles = []
    for det in detections:
        cx = det["cx"]
        if is_in_danger_zone(cx, frame_width):
            obstacles.append(det["label"])
    if not obstacles:
        return True, "The path ahead looks clear."
    unique = list(dict.fromkeys(obstacles))
    if len(unique) == 1:
        return False, f"There is a {unique[0]} directly ahead. Be careful."
    items = ", ".join(unique[:-1]) + f" and {unique[-1]}"
    return False, f"{items} are blocking your path. Please be careful."


# ─────────────────────────────────────────────
# Darius Mk.1 — Main App
# ─────────────────────────────────────────────
class DariusMk1:

    INTRO_TEXT = (
        "I am Darius, your smart vision assistant. "
        "I use a camera to detect objects, people, and obstacles around you. "
        "I can guide you while walking, read signs, and alert you to dangers. "
        "Say start or start trip whenever you are ready to begin."
    )

    def __init__(self):
        print("[INFO] Initializing Darius Mk.1...")

        # Model
        print("[INFO] Loading YOLOv8n...")
        self.model = YOLO("yolov8n.pt")
        print("[INFO] Model ready.")

        # State
        self.state = State.IDLE
        self.state_lock = threading.Lock()

        # Detection data
        self.last_detections = []
        self.last_frame = None
        self.frame_lock = threading.Lock()

        # Presence tracking for alerts
        self.objects_in_danger_zone = set()
        self.alerted_objects = set()

        # Detection timing — every 2 seconds
        self.last_detection_time = 0.0
        self.detection_interval = 2.0

        # Voice + listener
        self.voice = VoiceEngine()
        self.listener = Listener(callback=self.on_speech)
        self.listener.start()

        # Camera
        self.cap = None
        self._running = False

        print("[INFO] Darius ready.\n")

    # ─────────────────────────────────────────
    # Speech Router — called on every heard phrase
    # ─────────────────────────────────────────
    def on_speech(self, text: str):
        words = set(text.lower().split())
        state = self.state

        # "end" works in ANY state
        if "end" in words or ("stop" in words and "trip" in words):
            self.transition(State.END)
            return

        # Ignore all input while Darius is mid-introduction
        if state == State.INTRODUCING:
            print(f"[INFO] Introducing — ignored: '{text}'")
            return

        if state == State.IDLE:
            if any(w in words for w in ["introduce", "introduction", "yourself", "about"]):
                threading.Thread(target=self.do_introduce, daemon=True).start()
            elif any(w in words for w in ["start", "trip", "begin", "go", "walk"]):
                threading.Thread(target=self.do_overview, daemon=True).start()
            else:
                print(f"[INFO] In IDLE — ignored: '{text}'")

        elif state == State.OVERVIEW:
            # Waiting for yes/no after "Are you ready to walk?"
            if any(w in words for w in ["yes", "ready", "sure", "okay", "ok", "go", "let's", "lets"]):
                self.transition(State.WALKING)
                self.voice.speak("Great! Starting live guidance. I will alert you to any obstacles.", priority=True)
            elif any(w in words for w in ["no", "not", "wait", "hold"]):
                self.voice.speak("Okay, take your time. Say start whenever you are ready.", priority=True)
                self.transition(State.IDLE)

        elif state == State.WALKING:
            self.handle_walking_command(words, text)

    def handle_walking_command(self, words, text):
        """Handle on-demand voice queries while walking."""
        # Describe surroundings
        if len({"what", "see", "around", "describe", "visible"} & words) >= 2:
            threading.Thread(target=self.cmd_describe, daemon=True).start()
        # Count people
        elif len({"how", "many", "people", "count", "persons"} & words) >= 2:
            threading.Thread(target=self.cmd_count_people, daemon=True).start()
        # Path check
        elif len({"path", "clear", "safe", "ahead", "obstacle", "way"} & words) >= 2:
            threading.Thread(target=self.cmd_path_clear, daemon=True).start()
        # Read signs
        elif len({"read", "sign", "signs", "text", "board"} & words) >= 2:
            threading.Thread(target=self.cmd_read_signs, daemon=True).start()

    # ─────────────────────────────────────────
    # State Transitions
    # ─────────────────────────────────────────
    def transition(self, new_state):
        with self.state_lock:
            print(f"[STATE] {self.state} → {new_state}")
            self.state = new_state

    # ─────────────────────────────────────────
    # State Handlers
    # ─────────────────────────────────────────
    def do_greet(self):
        """Called once at startup."""
        self.voice.speak_and_wait(
            "Hello! How are you doing today? "
            "I am Darius, your smart glasses assistant. "
            "Say introduce to know more about me, or say start to begin your trip."
        )

    def do_introduce(self):
        self.transition(State.INTRODUCING)
        self.voice.speak_and_wait(self.INTRO_TEXT)
        self.voice.speak_and_wait("Say start whenever you are ready to begin your trip.")
        self.transition(State.IDLE)

    def do_overview(self):
        """Scan surroundings and give an overview before walking."""
        self.transition(State.OVERVIEW)
        self.voice.speak_and_wait("Sure! Let me take a look at your surroundings first.")
        time.sleep(1.5)  # give camera a moment to settle

        with self.frame_lock:
            detections = self.last_detections[:]
            frame = self.last_frame

        # Describe what's visible
        scene = describe_detections(detections)
        self.voice.speak_and_wait(scene)

        # Path status
        if frame is not None:
            clear, path_msg = path_status(detections, frame.shape[1])
            self.voice.speak_and_wait(path_msg)
        else:
            self.voice.speak_and_wait("I couldn't fully assess the path yet.")

        time.sleep(0.5)
        self.voice.speak_and_wait("Are we ready to walk?")

    # ─────────────────────────────────────────
    # On-demand Commands (during WALKING)
    # ─────────────────────────────────────────
    def cmd_describe(self):
        with self.frame_lock:
            detections = self.last_detections[:]
        self.voice.speak(describe_detections(detections), priority=True)

    def cmd_count_people(self):
        with self.frame_lock:
            detections = self.last_detections[:]
        count = sum(1 for d in detections if d["label"] == "person")
        if count == 0:
            self.voice.speak("I don't see any people around you.", priority=True)
        elif count == 1:
            self.voice.speak("There is 1 person nearby.", priority=True)
        else:
            self.voice.speak(f"There are {count} people around you.", priority=True)

    def cmd_path_clear(self):
        with self.frame_lock:
            detections = self.last_detections[:]
            frame = self.last_frame
        if frame is None:
            self.voice.speak("I can't check right now.", priority=True)
            return
        _, msg = path_status(detections, frame.shape[1])
        self.voice.speak(msg, priority=True)

    def cmd_read_signs(self):
        if not OCR_AVAILABLE:
            self.voice.speak("Sign reading is not available. Please install pytesseract.", priority=True)
            return
        with self.frame_lock:
            frame = self.last_frame
        if frame is None:
            self.voice.speak("No frame available.", priority=True)
            return
        self.voice.speak("Checking for signs.", priority=True)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = pytesseract.image_to_string(gray, config="--psm 11")
            words = [w.strip() for w in text.split() if len(w.strip()) > 2][:6]
            if words:
                self.voice.speak("I can see: " + " ".join(words), priority=True)
            else:
                self.voice.speak("I don't see any readable signs.", priority=True)
        except Exception as e:
            print(f"[ERROR] OCR: {e}")
            self.voice.speak("Couldn't read signs right now.", priority=True)

    # ─────────────────────────────────────────
    # Presence-Based Danger Zone Alerts
    # ─────────────────────────────────────────
    def update_danger_zone(self, current_in_zone: set, detections, frame_width):
        left = self.objects_in_danger_zone - current_in_zone
        for label in left:
            self.alerted_objects.discard(label)

        newly_entered = current_in_zone - self.alerted_objects
        self.objects_in_danger_zone = current_in_zone

        for label in newly_entered:
            self.alerted_objects.add(label)
            for det in detections:
                if det["label"] == label and is_in_danger_zone(det["cx"], frame_width):
                    direction = "left" if det["cx"] > frame_width / 2 else "right"
                    self.voice.speak(f"{label} ahead. Move {direction}.")
                    break

    # ─────────────────────────────────────────
    # Detection (runs every 2 seconds)
    # ─────────────────────────────────────────
    def run_detection(self, frame):
        results = self.model.predict(
            frame,
            conf=0.5,
            imgsz=320,
            verbose=False,
            stream=True
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = (x1 + x2) / 2
                detections.append({
                    "label": label,
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "cx": cx
                })
        return detections

    # ─────────────────────────────────────────
    # Draw HUD
    # ─────────────────────────────────────────
    def draw(self, frame, detections):
        w = frame.shape[1]
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = det["cx"]
            in_danger = is_in_danger_zone(cx, w)
            color = (0, 0, 255) if in_danger else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, det["label"], (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Danger zone lines
        cv2.line(frame, (int(w * 0.33), 0), (int(w * 0.33), frame.shape[0]), (100, 100, 255), 1)
        cv2.line(frame, (int(w * 0.66), 0), (int(w * 0.66), frame.shape[0]), (100, 100, 255), 1)

        # State HUD
        state_colors = {
            State.IDLE:        (200, 200, 0),
            State.INTRODUCING: (200, 100, 255),
            State.OVERVIEW:    (0, 200, 255),
            State.WALKING:     (0, 255, 0),
            State.END:         (0, 0, 255),
        }
        color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(frame, f"State: {self.state}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, "Darius Mk.1 | Say 'end' to stop", (8, frame.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return frame

    # ─────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────
    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera.")
            return

        self._running = True

        # Greet the user in a background thread so camera starts immediately
        threading.Thread(target=self.do_greet, daemon=True).start()

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            now = time.time()

            # Always run detection every 2 seconds (needed for overview + walking)
            # But only run alerts if in WALKING state
            if now - self.last_detection_time >= self.detection_interval:
                detections = self.run_detection(frame)
                self.last_detection_time = now

                with self.frame_lock:
                    self.last_detections = detections
                    self.last_frame = frame.copy()

                # Only give automatic alerts when WALKING
                if self.state == State.WALKING:
                    current_in_zone = {
                        det["label"] for det in detections
                        if is_in_danger_zone(det["cx"], frame.shape[1])
                    }
                    self.update_danger_zone(current_in_zone, detections, frame.shape[1])

            # Draw detections on frame
            with self.frame_lock:
                draw_dets = self.last_detections[:]
            frame = self.draw(frame, draw_dets)
            cv2.imshow("Darius Mk.1", frame)

            # Check for END state or Q key
            if self.state == State.END or (cv2.waitKey(1) & 0xFF == ord("q")):
                break

        # Shutdown
        if self.state != State.END:
            self.transition(State.END)
        self.voice.speak_and_wait("Goodbye! Stay safe.")
        self.cap.release()
        cv2.destroyAllWindows()
        self.listener.stop()
        self.voice.stop()
        print("[INFO] Darius offline.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    darius = DariusMk1()
    darius.run()