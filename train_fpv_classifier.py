import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Paths
train_dir = 'data/fpv_dataset/train'
val_dir   = 'data/fpv_dataset/val'

# Data transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets
train_ds = datasets.ImageFolder(train_dir, transform=transform)
val_ds   = datasets.ImageFolder(val_dir, transform=transform)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl   = torch.utils.data.DataLoader(val_ds, batch_size=16)

# Model
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Training Loss: {total_loss/len(train_dl):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")

# Save trained model
torch.save(model.state_dict(), 'models/fpv_classifier.pth')
print("✅ Model saved to models/fpv_classifier.pth")
