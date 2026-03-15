import torch
from PIL import Image
from torchvision.transforms import v2

from BirdClassification import InsertModelNameThatWeLoadedIn

model = MyModel()
model.load_state_dict(torch.load("model.pt", weights_only=True))

transforms = v2.Compose([
    v2.ToTensor(),
    v2.Resize((96,96))
])

img = Image.open("demo_peacock.png").convert('RGB')
img = transforms(img)

img = torch.unsqueeze(img, 0)

pred = model(img)
print(pred.item())

