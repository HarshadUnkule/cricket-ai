import cv2, os

video_path = 'data/raw_videos/match1_2min.mp4'
output_dir = 'data/frames/'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
print("Frame Rate:", fps)

i = 0
success, frame = cap.read()
while success:
    filename = os.path.join(output_dir, f"frame_{i:05d}.jpg")
    cv2.imwrite(filename, frame)
    success, frame = cap.read()
    i += 1

cap.release()
print(f"✅ Extracted {i} frames at {fps} FPS (full motion captured!)")
