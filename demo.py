import torch
from PIL import Image
from torchvision.transforms import v2

from BirdClassification import ConvNet

model = ConvNet()
model.load_state_dict(torch.load("birdclassification_ver1.pt", weights_only=True))

model.eval()

transforms = v2.Compose([
    v2.Resize((96,96)),
    v2.ToTensor()
    
])

img = Image.open("demo_barbet.png").convert('RGB')
img = transforms(img)

img = torch.unsqueeze(img,0)

# Source for printing class predictions: https://stackoverflow.com/questions/58111456/how-to-print-out-the-correct-predicted-category

Classes = [
    "Asian Green Bee-Eater",
    "Brown-Headed Barbet",
    "Cattle Egret",
    "Common Kingfisher",
    "Common Myna",
    "Common Rosefinch",
    "Common Tailorbird",
    "Coppersmith Barbet",
    "Foreset Wagtail",
    "Gray Wagtail",
    "Hoopoe",
    "House Crow",
    "Indian Grey Hornbill",
    "Indian Peacock",
    "Indian Pitta",
    "Indian Roller",
    "Jungle Babbler",
    "Northern Lapwing",
    "Red-Wattled Lapwing",
    "Ruddy Shelduck",
    "Rufous Treepie",
    "Sarus Crane",
    "White Wagtail",
    "White-Breasted Kingfisher",
    "White-Breasted Waterhen"
]

pred = model(img)
pred = torch.argmax(pred,dim=1)

print(pred.item())
print(Classes[pred.item()])
