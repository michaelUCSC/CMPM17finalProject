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
import wandb
torch.manual_seed(10)
run = wandb.init(project="Indian Bird Classification", name="Runs_2")

root = "Data"

#Transforms:
transforms = v2.Compose([
    
    v2.ToTensor(), 
    v2.Resize((96,96)),
    v2.RandomVerticalFlip(0.5),
    v2.RandomHorizontalFlip(0.5),
    v2.RandomPerspective(0.5),
    v2.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.5, hue=0.1),
    v2.RandomInvert(0.1)
])

test_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((96,96))
])

val_transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((96,96))
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

# print(random_images)

# plt.figure(figsize=(20, 5))
# for idx, fp in enumerate(random_images, start=1):
#     img = Image.open(fp)
#     plt1 = plt.subplot(5,20, idx)
#     plt1.imshow(img)
#     plt1.set_title(os.path.basename(os.path.dirname(fp)))
#     plt1.axis('off')

# plt.tight_layout()
# plt.show()

# Convolution Layers

#Change channels in conv if model is really bad.

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,4,3,1,1)
        self.conv2 = nn.Conv2d(4,3,3,1,1)
        self.conv3 = nn.Conv2d(3,12,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(12*12*12,50)
        self.fc2 = nn.Linear(50,25)
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

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

NUM_EPOCHS = 5

for epochs in range (1, NUM_EPOCHS+1):
#added +1 to num_epochs because the loop starts from 1 instead of 0
    avgLossInEpoch = 0
    num_batches = 0
    correct_predictions = 0
    total_samples = 0
    print("------------Training------------\nPlease wait (should take a minute or two)...")
    for x_batch, y_batch in train_loader:
        num_batches += 1

        train_pred = model(x_batch)
        train_loss = loss_function(train_pred, y_batch)

        avgLossInEpoch += train_loss


        predicted_classes = train_pred.argmax(dim=1)
        correct_predictions += (predicted_classes == y_batch).sum().item()
        total_samples += y_batch.size(0)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(train_loss.item())
        run.log({"Train Loss":train_loss})
    avgLossInEpoch /= num_batches
    accuracy = correct_predictions / total_samples

    print(f"Epoch {epochs}")
    print(f"Average Loss: {avgLossInEpoch:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    avgValLoss = 0
    print("------------Validation------------")
    for x_batch, y_batch in val_loader:
        val_pred = model(x_batch)
        val_loss = loss_function(val_pred, y_batch)

        predicted_classes = val_pred.argmax(dim=1)
        correct_predictions += (predicted_classes == y_batch).sum().item()
        total_samples += y_batch.size(0)

        avgValLoss += val_loss
        print(val_loss.item())
        run.log({"Validation Loss":val_loss})
    avgValLoss /= num_batches
    print(f"Average validation loss in epoch {epochs}: {avgValLoss}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


total_samples = 0
avgLossInEpoch = 0
num_batches = 0
correct_predictions = 0
avgTestLoss = 0

print("------------Testing------------")
model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        test_pred = model(x_batch)
        loss = loss_function(test_pred, y_batch)

        predicted_classes = test_pred.argmax(dim=1)
        correct_predictions += (predicted_classes == y_batch).sum().item()
        total_samples += y_batch.size(0)

        avgTestLoss += loss
        print(loss.item())
    
    avgTestLoss /= num_batches
    print(f"Average testing loss in epoch {epochs}: {avgTestLoss}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# print("finished training, saving model")

# torch.save(model.state_dict(), "birdclassification_ver1.pt")