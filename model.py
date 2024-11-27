import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        #Fresh start
        self.conv1  = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1    = nn.BatchNorm2d(8)
        self.conv2  = nn.Conv2d(8, 12, 3, padding=1)
        self.bn2    = nn.BatchNorm2d(12)
        self.pool1  = nn.MaxPool2d(2,2)

        self.conv3  = nn.Conv2d(12, 16, 3, padding=1)
        self.bn3    = nn.BatchNorm2d(16)
        self.conv4  = nn.Conv2d(16, 20, 3, padding=1)
        self.bn4    = nn.BatchNorm2d(20)
        self.pool2  = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(20*7*7, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.view(-1, 20*7*7)
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)