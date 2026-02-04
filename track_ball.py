from ultralytics import YOLO
import cv2
import numpy as np

# ---------------------------------------------------------
#                 SIMPLE KALMAN BALL TRACKER
# ---------------------------------------------------------
class KalmanBallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        # State: x, y, vx, vy (Constant Velocity Model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], np.float32)

        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.3

    def update(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])


# ---------------------------------------------------------
#                     LOAD YOLO MODEL
# ---------------------------------------------------------
model = YOLO(
    r"C:\Users\Prathmesh Bhosale\cricket_ai\runs\detect\ball_detector9\weights\best.pt"
)

# ---------------------------------------------------------
#                     LOAD VIDEO
# ---------------------------------------------------------
cap = cv2.VideoCapture(
    r"C:\Users\Prathmesh Bhosale\cricket_ai\data\raw_videos\match1.mp4"
)

tracker = KalmanBallTracker()
trajectory = []


# ---------------------------------------------------------
#                   MAIN PROCESSING LOOP
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    ball_detected = False
    bx, by = None, None

    # ---- Get ONLY the highest confidence ball detection ----
    best_score = 0
    best_box = None

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = box
        if int(cls) == 0 and score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)

    # ---- If we got a detection ----
    if best_box:
        x1, y1, x2, y2 = best_box
        bx = int((x1 + x2) / 2)
        by = int((y1 + y2) / 2)
        px, py = tracker.update(bx, by)
        ball_detected = True
    else:
        px, py = tracker.predict()

    # ---- Save and draw trajectory ----
    trajectory.append((px, py))

    # Draw ball point
    cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)

    # Draw clean trajectory line
    for i in range(1, len(trajectory)):
        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)

    cv2.imshow("Ball Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
