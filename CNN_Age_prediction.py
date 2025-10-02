import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import re

# Path to dataset
data_path = "dataset/utkface_aligned_cropped/UTKFace"

def extract_age(filename):
    match = re.match(r"(\d+)_", filename)
    return int(match.group(1)) if match else None

# Custom Dataset class
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.ages = [extract_age(f) for f in os.listdir(root_dir)]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        age = self.ages[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(age, dtype=torch.float32)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = UTKFaceDataset(data_path, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x.squeeze()

# Train function
def train_model(model, train_loader, test_loader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, ages in train_loader:
            images, ages = images.to(device), ages.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, ages)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for images, ages in test_loader:
                images, ages = images.to(device), ages.to(device)
                outputs = model(images)
                loss = criterion(outputs, ages)
                test_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Test Loss = {test_loss/len(test_loader):.4f}")

# Train CNN from scratch
cnn_model = CNNModel()
train_model(cnn_model, train_loader, test_loader, epochs=10)

# Fine-tune ResNet18
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 1)  # Change last FC layer
train_model(resnet18, train_loader, test_loader, epochs=10)
