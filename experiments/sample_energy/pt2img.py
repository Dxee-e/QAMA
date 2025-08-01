import torch
from torchvision import transforms
from PIL import Image

img = torch.load('img.pt')
print(img.shape)

img = img * 0.5 + 0.5
img = (img * 255).byte()

to_pil = transforms.ToPILImage()
pil_image = to_pil(img)

pil_image.save('image.png')