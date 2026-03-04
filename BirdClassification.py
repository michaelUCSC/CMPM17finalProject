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
root = "training_set"

#Transforms:
transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize(96,96),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomPerspective(0.5),
    v2.RandomChannelPermutation(),
    v2.RandomInvert(0.1)
])

# Take Indian Dataset and extract training_set with the data
# Make a new folder called "__pycache__"

#Train/Test/Val Split:
splitfolders.ratio(root,output= "__pycache__",
                   ratio = (0.7,0.15,0.15))

root = "__pycache__"

#Dataloaders for Train, Val, Test
train_data = ImageFolder(os.path.join(root,'train'),transform=transforms)
test_data = ImageFolder(os.path.join(root,'test'),transform=transforms)
val_data = ImageFolder(os.path.join(root,'val'),transform=transforms)

train_loader = DataLoader(train_data,batch_size=30,shuffle=True)
test_loader = DataLoader(test_data,batch_size=15,shuffle=False)
val_loader = DataLoader(val_data,batch_size=15,shuffle=False)


data = ImageFolder(root)

#Image Displays
import os
from PIL import Image
import random

base_dir = os.path.dirname(__file__)
p = os.path.abspath(os.path.join(base_dir, "training_set"))

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = []
for root, _, files in os.walk(p):
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