import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor, from_numpy, FloatTensor
import torch
import numpy as np
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class NNModelBase(nn.Module):
    def __init__(self, device):
        super(NNModelBase, self).__init__()
        self.name = 'Baseline'
        self.single_frame = True
        self.device = device
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2))
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.convdrop = nn.Dropout2d(0.2)

        self.conv1 = torch.nn.Conv2d(3, 6, (3, 5), padding=(1, 2)) # 108 x 96
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 16, (3, 3), padding=(1, 1)) # 54 x 96
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 5), padding=(1, 2)) # 54 x 48
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 48, (3, 3), padding=(1, 1)) # 27 x 48
        self.bn4 = torch.nn.BatchNorm2d(48)
        self.conv5 = torch.nn.Conv2d(48, 64, (3, 5), padding=(1, 2)) # 27 x 24
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.conv6 = torch.nn.Conv2d(64, 96, (3, 3), padding=(1, 1)) # 13 x 24
        self.bn6 = torch.nn.BatchNorm2d(96)
        self.conv7 = torch.nn.Conv2d(96, 128, (3, 3), padding=(1, 1)) # 13 x 12
        self.bn7 = torch.nn.BatchNorm2d(128)
        self.conv8 = torch.nn.Conv2d(128, 160, (3, 3), padding=(1, 1)) # 6 x 12
        self.bn8 = torch.nn.BatchNorm2d(160)
        self.conv9 = torch.nn.Conv2d(160, 192, (3, 3), padding=(1, 1)) # 6 x 6
        self.bn9 = torch.nn.BatchNorm2d(192)

        self.fc1 = nn.Linear(6912, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 60)

    def predict_img(self, x):

        x = self.convdrop(self.pool2(F.leaky_relu(self.bn1(self.conv1(x)))))
        x = self.convdrop(self.pool1(F.leaky_relu(self.bn2(self.conv2(x)))))
        x = self.convdrop(self.pool2(F.leaky_relu(self.bn3(self.conv3(x)))))
        x = self.convdrop(self.pool1(F.leaky_relu(self.bn4(self.conv4(x)))))
        x = self.convdrop(self.pool2(F.leaky_relu(self.bn5(self.conv5(x)))))
        x = self.convdrop(self.pool1(F.leaky_relu(self.bn6(self.conv6(x)))))
        x = self.convdrop(self.pool2(F.leaky_relu(self.bn7(self.conv7(x)))))
        x = self.convdrop(self.pool1(F.leaky_relu(self.bn8(self.conv8(x)))))
        x = self.convdrop(self.pool2(F.leaky_relu(self.bn9(self.conv9(x)))))

        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward(self, X: torch.Tensor):
        all_predictions = []
        if self.single_frame:
            for sample in X:
                res = self.predict_img(sample[10].unsqueeze(0))
                all_predictions.append(res.squeeze(0))
        else:
            for sample in X:
                predicted = torch.zeros(size=(1, 60)).to(self.device)
                for img in sample:
                    k = self.predict_img(img.unsqueeze(0))
                    predicted += k
                predicted = torch.div(predicted, 20)
                all_predictions.append(predicted.squeeze(0))
        all_predictions = torch.stack(all_predictions).to(self.device)
        return all_predictions
