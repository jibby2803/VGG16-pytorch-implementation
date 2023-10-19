import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

class AnimalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.image_paths = []
        self.labels = []
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        self.transform = transform
        
        if train:
            data_path = os.path.join(root, 'train')
        else:
            data_path = os.path.join(root, 'test')
        
        for i, category in enumerate(self.categories):
            data_files = os.path.join(data_path,category)
            for item in os.listdir(data_files):
                path = os.path.join(data_files,item)
                self.image_paths.append(path)
                self.labels.append(i)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        # image = torch.from_numpy(image)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # image = np.transpose(image, (2,0,1))
        return image, label

if __name__=='__main__':

    train_transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
    ])
    test_transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
    ]) 

    train_dataset = AnimalDataset(root='data', train=True, transform=train_transform)
    print(train_dataset.__len__())
    test_dataset = AnimalDataset(root='data', train=False, transform=test_transform)
    print(test_dataset.__len__())
    train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)
    
    # for images, labels in train_loader:
    #     print(images.shape, labels.shape)
    for images, labels in test_loader:
        print(images.shape, labels.shape)
        
    
        
        
                 
        
