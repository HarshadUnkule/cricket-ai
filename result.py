from ultralytics import YOLO
import cv2

model = YOLO(r"C:\Users\Prathmesh Bhosale\cricket_ai\runs\detect\ball_detector9\weights\best.pt")

img = cv2.imread(r"C:\Users\Prathmesh Bhosale\cricket_ai\data\raw_videos\frame_test.jpg")  # choose any frame
results = model(img)[0]

print(results.boxes.data)
