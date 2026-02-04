from ultralytics import YOLO
import cv2, os

# Load YOLOv8 pretrained on COCO
model = YOLO('yolov8n.pt')   # 'n' = nano (fast). You can use 's' for slightly more accurate.

input_dir = 'data/frames'
output_dir = 'data/fpv_frames_yolo'
os.makedirs(output_dir, exist_ok=True)

# Class names in COCO (important indices)
COCO_CLASSES = model.names
def name_to_id(name): return [k for k,v in COCO_CLASSES.items() if v == name][0]

# IDs for key cricket objects
PERSON_ID = name_to_id('person')
BAT_ID    = name_to_id('baseball bat')
BALL_ID   = name_to_id('sports ball')

for f in sorted(os.listdir(input_dir)):
    if not f.endswith('.jpg'): 
        continue

    frame_path = os.path.join(input_dir, f)
    results = model(frame_path, verbose=False)[0]
    detections = results.boxes.cls.tolist()

    # Count objects
    num_persons = detections.count(PERSON_ID)
    has_bat = BAT_ID in detections
    has_ball = BALL_ID in detections

    # Logic for FPV
    if num_persons >= 2 and has_bat:
        cv2.imwrite(os.path.join(output_dir, f), cv2.imread(frame_path))

print("✅ FPV frames saved to:", output_dir)
