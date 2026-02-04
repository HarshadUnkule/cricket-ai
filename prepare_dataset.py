import os
import shutil

BASE = r"C:\Users\Prathmesh Bhosale\cricket_ai\data"
SRC_LABELS = os.path.join(BASE, "ball_for_labeling")  # correct label folder

DATASET = os.path.join(BASE, "ball_dataset")
IMG_TRAIN = os.path.join(DATASET, "images", "train")
IMG_VAL = os.path.join(DATASET, "images", "val")
LBL_TRAIN = os.path.join(DATASET, "labels", "train")
LBL_VAL = os.path.join(DATASET, "labels", "val")

def check_split(img_dir, lbl_dir, name):
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    total = len(imgs)
    missing = []
    empty = []
    src_has = []

    for img in imgs:
        txt = img.replace(".jpg", ".txt")
        lbl_path = os.path.join(lbl_dir, txt)
        src_path = os.path.join(SRC_LABELS, txt)

        if os.path.exists(lbl_path):
            if os.path.getsize(lbl_path) == 0:
                empty.append(txt)
        else:
            missing.append(txt)

        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            src_has.append(txt)

    print(f"\n=== {name} ===")
    print("Images:", total)
    print("Missing labels:", len(missing))
    print("Empty labels:", len(empty))
    print("Labels in source folder:", len(src_has))
    return missing + empty

def copy_labels(txt_list, dest_dir):
    for txt in txt_list:
        src = os.path.join(SRC_LABELS, txt)
        dst = os.path.join(dest_dir, txt)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print("Copied:", txt)

if __name__ == "__main__":
    to_fix_train = check_split(IMG_TRAIN, LBL_TRAIN, "TRAIN")
    to_fix_val = check_split(IMG_VAL, LBL_VAL, "VAL")

    ans = input("Copy non-empty labels from ball_for_labeling? (y/N): ").strip().lower()
    if ans == 'y':
        print("\nFixing TRAIN...")
        copy_labels(to_fix_train, LBL_TRAIN)
        print("\nFixing VAL...")
        copy_labels(to_fix_val, LBL_VAL)
        print("\nDONE – re-run training!")
