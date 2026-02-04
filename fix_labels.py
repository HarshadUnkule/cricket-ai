import os

folders = [
    r"C:\Users\Prathmesh Bhosale\cricket_ai\data\ball_for_labeling",
    r"C:\Users\Prathmesh Bhosale\cricket_ai\data\ball_dataset\labels\train",
    r"C:\Users\Prathmesh Bhosale\cricket_ai\data\ball_dataset\labels\val"
]

for folder in folders:
    print("\nFixing:", folder)
    for f in os.listdir(folder):
        if f.endswith(".txt"):
            path = os.path.join(folder, f)
            lines = open(path).read().strip().splitlines()

            new_lines = []
            for line in lines:
                parts = line.split()
                parts[0] = "0"   # FORCE class 0
                new_lines.append(" ".join(parts))

            open(path, "w").write("\n".join(new_lines))

            print("Fixed:", f)

print("\nDONE — ALL LABELS NOW CLASS 0")
