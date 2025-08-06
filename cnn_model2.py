import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
image_dir = './imagesOrg/1024x768'
metadata_file = './metadata.csv'

# Load metadata
df = pd.read_csv(metadata_file, dtype={'image_id': str})

# Custom Dataset
class IQADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id'].strip()
        possible_filenames = [f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png"]

        img_path = None
        for fname in possible_filenames:
            full_path = os.path.join(self.image_dir, fname)
            if os.path.exists(full_path):
                img_path = full_path
                break

        if img_path is None:
            raise FileNotFoundError(f"No image found for ID: {image_id}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        mos = float(row['MOS'])
        return image, cnn_model2.tensor(mos, dtype=cnn_model2.float32)

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = IQADataset(train_df, image_dir, transform)
test_dataset = IQADataset(test_df, image_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# CNN Model
class CNNRegressor(nn.Module):
    def __init__(self):
        super(CNNRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

# Instantiate model, loss, optimizer
model = CNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

# Metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"\nEvaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot predictions
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("True MOS")
plt.ylabel("Predicted MOS")
plt.title("PyTorch CNN: Predicted vs True MOS")
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.grid()
plt.show()

# Save model
torch.save(model.state_dict(), "cnn_torch_model.pth")
