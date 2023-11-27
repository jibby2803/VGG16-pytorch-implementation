from vgg16_model import VGG16_net
import cv2
from transforms import *
import torch


model = VGG16_net()
model.load_state_dict(torch.load('model/best.pt'))

def predict(img):
    img = test_transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        predictions = torch.argmax(outputs, dim=1)
        return predictions.item()
    
img = cv2.imread('data/test/cat/cat_0011.jpg')
print(predict(img))



