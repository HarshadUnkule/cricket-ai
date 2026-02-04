import torch
from torchvision import models, transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm

# Paths
frames_dir = 'data/frames'
model_path = 'models/fpv_classifier.pth'
output_dir = 'data/fpv_sorted'

# Create output folders
os.makedirs(os.path.join(output_dir, 'FPV'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'NonFPV'), exist_ok=True)

# Load model
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class names (same order as your training folders)
classes = ['FPV', 'NonFPV']

# Inference loop
for frame in tqdm(os.listdir(frames_dir), desc="Classifying frames"):
    frame_path = os.path.join(frames_dir, frame)

    try:
        image = Image.open(frame_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, pred = torch.max(outputs, 1)
            label = classes[pred.item()]

        # Move frame to respective folder
        dest_path = os.path.join(output_dir, label, frame)
        shutil.copy(frame_path, dest_path)
    except Exception as e:
        print(f"Error processing {frame}: {e}")

print("\n✅ All frames classified and sorted into data/fpv_sorted/FPV and NonFPV")
