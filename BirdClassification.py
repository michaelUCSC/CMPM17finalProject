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

root = "IndianBird_Dataset/training_set/training_set"

transforms = v2.Compose([
    v2.ToTensor(),
    v2.RandomResize(40,100),
    v2.RandomHorizontalFlip([0.5]),
    v2.RandomVerticalFlip([0.5]),
    v2.RandomPerspective([0.5]),
    v2.RandomChannelPermutation(),
    v2.RandomInvert(0.1)
])

splitfolders.ratio(root,outputs="IndianBird_Dataset/training_set/training_set",
                   ratio = (0.7,0.15,0.15))

data = ImageFolder(root)
