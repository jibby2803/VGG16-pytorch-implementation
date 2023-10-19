import torch
import torch.nn as nn
import torch.nn.functional as F  

class VVG19_net(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = self.conv_block(in_channels=self.in_channels, block=[64, 64])
        self.conv2 = self.conv_block(in_channels=64, block=[128, 128])
        self.conv3 = self.conv_block(in_channels=128, block=[256, 256, 256, 256])
        self.conv4 = self.conv_block(in_channels=256, block=[512, 512, 512, 512])
        self.conv5 = self.conv_block(in_channels=512, block=[512, 512, 512, 512])
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fcs(x)
        return x
    
    def conv_block(self, in_channels ,block):
        layers = []
        for i in block:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                       nn.BatchNorm2d(i),
                       nn.ReLU()]
            in_channels = i
        layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        return nn.Sequential(*layers)
        
if __name__=='__main__':
    model = VVG19_net()
    x = torch.randn(8, 3, 224, 224)
    print(model)
    print(model(x).shape)
