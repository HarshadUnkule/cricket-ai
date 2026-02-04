from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Users\Prathmesh Bhosale\cricket_ai\data\ball_dataset\ball.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="ball_detector"
)
