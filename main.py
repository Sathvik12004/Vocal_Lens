import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize model, speech engine, and webcam
model = YOLO('yolov8n.pt')  # use yolov8n.pt for CPU-friendly speed
engine = pyttsx3.init()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_width = int(cap.get(3))

spoken = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]
    
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if conf < 0.5:
            continue

        # Draw box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Suggest direction if object in front
        center_x = (x1 + x2) / 2
        if frame_width * 0.33 < center_x < frame_width * 0.66 and label not in spoken:
            direction = "left" if center_x > frame_width / 2 else "right"
            msg = f"Warning: {label} ahead. Move to the {direction}."
            print(msg)
            engine.say(msg)
            engine.runAndWait()
            spoken.add(label)

    cv2.imshow("Blind Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()