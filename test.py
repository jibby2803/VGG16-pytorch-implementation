import torch
from vgg16_model import VGG16_net
import matplotlib.pyplot as plt
import os
# from predict import predict
import cv2
import numpy as np
from transforms import test_transform

categories = ["butterfly", "cat", "chicken", "cow", "dog",
              "elephant", "horse", "sheep", "spider", "squirrel"]

device = device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

test_model = VGG16_net().to(device)
test_model.load_state_dict(torch.load('./model/best.pt', map_location=torch.device('cpu')))
test_model.eval()

image_list = os.listdir('./example')
# print(image_list)
# exit()

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i in range(10):
    org_img = cv2.imread(os.path.join('./example', image_list[i]))
    org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    # print(img)
    # exit()
    img = org_img.copy()
    img = test_transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = test_model(img)
        prediction = torch.argmax(output, dim=1)
        prediction = prediction.item()
        
    axes[i].imshow(org_img)  
    title = f"{categories[prediction]}"

    # if labels is not None:
    #     title += f"\nTrue Label: {categories[labels[i]]}"

    axes[i].set_title(title)
    axes[i].axis('off')

plt.tight_layout()
plt.show()