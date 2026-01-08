import os
import glob
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

from model import SimpleCNN, IMG_SIZE

def detect_roi(data_dir):
    """Detect Blue ROI from the first available *Propaint.jpg file"""
    search_pattern = os.path.join(data_dir, "**", "*Propaint.jpg")
    paint_files = glob.glob(search_pattern, recursive=True)
    
    if not paint_files:
        print("Error: No *Propaint.jpg files found in data directory.")
        return None

    ref_img_path = paint_files[0]
    print(f"Detecting ROI from: {ref_img_path}")
    
    img = cv2.imread(ref_img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Blue mask (adjust if needed)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No blue area found.")
        return None
        
    # Find largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    # Start slightly wider for robustness
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w += margin * 2
    h += margin * 2
    
    print(f"ROI Detected: x={x}, y={y}, w={w}, h={h}")
    return {"x": x, "y": y, "w": w, "h": h}

class CoinDataset(Dataset):
    def __init__(self, data_dir, roi, transform=None):
        self.data_dir = data_dir
        self.roi = roi
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load 0 (No Coin) - Folder "o"
        for f in glob.glob(os.path.join(data_dir, "o", "*Pro.jpg")):
            self.image_paths.append(f)
            self.labels.append(0)
            
        # Load 1 (Coin) - Folder "1"
        for f in glob.glob(os.path.join(data_dir, "1", "*Pro.jpg")):
            self.image_paths.append(f)
            self.labels.append(1)
            
        print(f"Loaded {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop ROI
        x, y, w, h = self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"]
        img = img[y:y+h, x:x+w]
        
        # Convert to PIL for transforms
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def train(data_dir, output_model_path, output_roi_path):
    # 1. Try to load existing manual ROI first
    roi = None
    if os.path.exists(output_roi_path):
        try:
            with open(output_roi_path, 'r') as f:
                roi = json.load(f)
            print(f"Using Manual ROI from {output_roi_path}: {roi}")
        except:
            print("Error loading ROI file. Falling back to auto-detection.")
            roi = None

    # 2. Fallback to Auto Detection
    if not roi:
        roi = detect_roi(data_dir)
    
    if not roi:
        return

    # Save ROI (if it was auto-detected or just to be safe)
    with open(output_roi_path, 'w') as f:
        json.dump(roi, f)

    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # Random rotation & shift
        transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Random brightness
        transforms.ToTensor(),
    ])

    dataset = CoinDataset(data_dir, roi, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}, Acc: {100 * correct / total:.2f}%")
        
    # Save Model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data directory (containing '1' and 'o')")
    parser.add_argument("--output_name", required=True, help="Base name for output model (e.g., 'model_A')")
    args = parser.parse_args()
    
    model_path = f"{args.output_name}.pth"
    roi_path = f"{args.output_name}_roi.json"
    
    train(args.data_dir, model_path, roi_path)
