from ultralytics import YOLO
import cv2
import os

MODEL_PATH = r"runs/detect/ball_detector9/weights/best.pt"
IMG_DIR = "perfect_dataset/images/train"
LBL_DIR = "perfect_dataset/labels/train"

os.makedirs(LBL_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

for img_name in sorted(os.listdir(IMG_DIR)):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)

    results = model(img, conf=0.15, verbose=False)[0]

    best = None
    best_conf = 0

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if int(cls) == 0 and conf > best_conf:
            best_conf = conf
            best = (x1, y1, x2, y2)

    label_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    if best:
        h, w, _ = img.shape
        x1, y1, x2, y2 = best
        xc = ((x1 + x2) / 2) / w
        yc = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h

        with open(label_path, "w") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        open(label_path, "w").close()

print("✅ Auto-labeling done. Now MANUALLY FIX boxes.")
