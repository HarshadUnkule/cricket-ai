import os
import shutil

# ----------- CONFIG -----------
FPV_FRAMES_DIR = r"C:\Users\Prathmesh Bhosale\cricket_ai\data\fpv_sorted\FPV"

OUT_TRAIN = "data/ball_dataset/images/train"
OUT_VAL = "data/ball_dataset/images/val"

FRAMES_PER_DELIVERY = 6
TRAIN_RATIO = 0.85  # 85% train, 15% val
# --------------------------------

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)


# 1) READ ALL FRAMES AND SORT THEM
frames = sorted(
    [f for f in os.listdir(FPV_FRAMES_DIR) if f.endswith(".jpg")],
    key=lambda x: int("".join(filter(str.isdigit, x)))
)

# Convert to integer list of frame numbers
frame_nums = [int("".join(filter(str.isdigit, f))) for f in frames]


# 2) GROUP FRAMES INTO DELIVERIES
deliveries = []
current = [frame_nums[0]]

for i in range(1, len(frame_nums)):
    # New delivery when gap > 5 frames
    if frame_nums[i] - frame_nums[i - 1] > 5:
        deliveries.append(current)
        current = []
    current.append(frame_nums[i])

if current:
    deliveries.append(current)

print(f"Detected {len(deliveries)} deliveries.")


# 3) PICK 6 REPRESENTATIVE FRAMES FROM EACH DELIVERY
selected_frames = []

for delivery in deliveries:
    if len(delivery) < FRAMES_PER_DELIVERY:
        continue  # skip very small segments

    start = delivery[0]

    picks = [
        start,
        start + 5,
        start + 10,
        start + 15,
        start + 20,
        start + 25
    ]

    # Only keep valid frames actually in delivery
    picks = [p for p in picks if p in delivery]

    # If fewer than 6, pad with equally spaced items
    if len(picks) < FRAMES_PER_DELIVERY:
        gap = len(delivery) // FRAMES_PER_DELIVERY
        extra = [delivery[i * gap] for i in range(FRAMES_PER_DELIVERY)]
        picks = extra

    selected_frames.append(picks)


# Flatten list
flat_picks = [f for group in selected_frames for f in group]

print(f"Total frames selected: {len(flat_picks)}")


# 4) SPLIT INTO TRAIN / VAL & COPY FILES
cutoff = int(len(flat_picks) * TRAIN_RATIO)

for i, frame_num in enumerate(flat_picks):
    frame_name = f"frame_{frame_num:05d}.jpg"
    src = os.path.join(FPV_FRAMES_DIR, frame_name)

    if not os.path.exists(src):
        continue

    if i < cutoff:
        dst = OUT_TRAIN
    else:
        dst = OUT_VAL

    shutil.copy(src, dst)

print("\n✅ DONE!")
print(f"Train images: {cutoff}")
print(f"Val images: {len(flat_picks) - cutoff}")
