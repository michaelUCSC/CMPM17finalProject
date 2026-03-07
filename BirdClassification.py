import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
import splitfolders
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import os
root = "Data"

#Transforms:
transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(96,96),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomPerspective(0.5),
    v2.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.1),
    v2.RandomInvert(0.1)
])

test_transforms = v2.Compose([
    v2.ToTensor()
])

val_transforms = v2.Compose([
    v2.ToTensor()
])

# Take Indian Dataset and extract training_set with the data

import os
from PIL import Image
import random

base_dir = os.path.dirname(__file__)
upper_dir = os.path.dirname(base_dir)
data_path = os.path.abspath(os.path.join(upper_dir, "Data"))

#Dataloaders for Train, Val, Test
train_data = ImageFolder(data_path, transform=transforms)
test_data = ImageFolder(data_path,transform=test_transforms)
val_data = ImageFolder(data_path,transform=val_transforms)

train_loader = DataLoader(train_data,batch_size=70,shuffle=True)
test_loader = DataLoader(test_data,batch_size=32,shuffle=False)
val_loader = DataLoader(val_data,batch_size=32,shuffle=False)

#Image Displays

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = []
for root, _, files in os.walk(data_path):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_ext:
            image_paths.append(os.path.join(root, filename))

num_to_show = min(100, len(image_paths))
random_images = random.sample(image_paths, k=num_to_show)

print(random_images)

plt.figure(figsize=(20, 5))
for idx, fp in enumerate(random_images, start=1):
    img = Image.open(fp)
    plt1 = plt.subplot(5,20, idx)
    plt1.imshow(img)
    plt1.set_title(os.path.basename(os.path.dirname(fp)))
    plt1.axis('off')

plt.tight_layout()
plt.show()

# Convolution Layers

#Change channels in conv if model is really bad.

class ConvNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3,6,3,1,1)
        self.conv2 = nn.Conv2d(6,16,3,1,1)
        self.conv3 = nn.Conv2d(16,48,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(12*12*48,50)
        self.fc2 = nn.Linear(400,5)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output


model = ConvNet()
model.train()

# Training Loop

# Try 50 epochs
# Testing loss doesn't matter at the moment, but try to keep losses on the low side so it's easier to fix later on.
# For each epoch, include validation.

# Maybeee include test just in case?? I don't know if it's required but should write just in case maybe.

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

NUM_EPOCHS = 50


for epochs in range (NUM_EPOCHS):
    print("------------Training------------")
    for x_batch, y_batch in train_loader:
        #Placeholder
    
    print("------------Validation------------")
    for x_batch, y_batch in val_loader:
        #Placeholder

print("------------Testing------------")
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
    #Placeholder