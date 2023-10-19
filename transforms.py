from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage, RandomResizedCrop, RandomHorizontalFlip

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
size = 224

train_transform = Compose([
                ToPILImage(),
                RandomResizedCrop(224, scale=(0.5,1.0)),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean, std)
            ])
test_transform = Compose([
                ToPILImage(),
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean, std)
            ])

