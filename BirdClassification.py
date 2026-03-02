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
    v2.RandomResize(40,100),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomVerticalFlip(0.5),
    v2.RandomPerspective(0.5),
    v2.RandomChannelPermutation(),
    v2.RandomInvert(0.1)
])

# Take Indian Dataset and extract training_set with the data
# Make a new file called "__pycache__"

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
p = os.path.abspath(os.path.join(base_dir, "..", "training_set"))

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = []
for root, _, files in os.walk(p):
    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_ext:
            image_paths.append(os.path.join(root, filename))

num_to_show = min(100, len(image_paths))
random_images = random.sample(image_paths, k=num_to_show)

plt.figure(figsize=(20, 5))
for idx, fp in enumerate(random_images, start=1):
    img = Image.open(fp)
    plt1 = plt.subplot(5, 20, idx)
    plt1.imshow(img)
    plt1.set_title(os.path.basename(os.path.dirname(fp)))
    plt1.axis('off')

plt.tight_layout()
plt.show()

# We were able to get matplotlib to show up, but the pictures will not show up.