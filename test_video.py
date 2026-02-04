from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO(r"C:\Users\Prathmesh Bhosale\cricket_ai\runs\detect\ball_detector9\weights\best.pt")

# CHOOSE WHICH VIDEO TO TEST
VIDEO_PATH = r"C:\Users\Prathmesh Bhosale\cricket_ai\data\raw_videos\match1_2min.mp4"
# VIDEO_PATH = r"C:\Users\Prathmesh Bhosale\cricket_ai\data\raw_videos\match1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

# Output video save path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("detected_output.mp4", fourcc, 30,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection (reduce confidence if ball is tiny)
    results = model.predict(frame, conf=0.15)

    # Draw boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.putText(frame, f"Ball {conf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Ball Detection Output", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
