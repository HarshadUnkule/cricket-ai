import os
import shutil

# ==============================
# CONFIG — CHANGE ONLY THIS
# ==============================
FRAMES_DIR = "data/frames"        # extracted frames
OUT_BASE = "perfect_dataset"     # output dataset folder

START_FRAME = 320    # first frame of delivery
END_FRAME   = 380    # last frame of delivery
# ==============================


IMG_OUT = os.path.join(OUT_BASE, "images", "train")
LBL_OUT = os.path.join(OUT_BASE, "labels", "train")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

count = 0

for i in range(START_FRAME, END_FRAME + 1):
    img_name = f"frame_{i:05d}.jpg"
    src_img = os.path.join(FRAMES_DIR, img_name)

    if not os.path.exists(src_img):
        print(f"⚠ Missing frame: {img_name}")
        continue

    # Copy image
    shutil.copy(src_img, IMG_OUT)

    # Create empty label file (YOLO format)
    label_name = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(LBL_OUT, label_name)

    if not os.path.exists(label_path):
        open(label_path, "w").close()

    count += 1

print("\n✅ PERFECT DATASET CREATED")
print(f"Frames copied: {count}")
print(f"From frame {START_FRAME} to {END_FRAME}")
print("➡ Now label EVERY frame manually")
