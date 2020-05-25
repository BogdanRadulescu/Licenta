import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor, from_numpy, FloatTensor
import torch
import numpy as np
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class NNModelBase(nn.Module):
    def __init__(self, single_frame=False):
        super(NNModelBase, self).__init__()
        self.name = 'Baseline'
        self.single_frame = single_frame
        self.pool = nn.MaxPool2d(2, 2)

        self.conv_4_1 = nn.Conv2d(3, 6, (2, 3))
        self.conv_4_2 = nn.Conv2d(6, 10, (2, 3))
        self.conv_4_3 = nn.Conv2d(10, 15, (2, 3))
        self.conv_4_4 = nn.Conv2d(15, 22, (2, 3))
        self.conv_4_5 = nn.Conv2d(22, 36, (2, 3))
        self.conv_4_6 = nn.Conv2d(36, 56, (2, 3))

        self.conv_4_7 = nn.Conv2d(56, 76, (2, 3))
        self.conv_4_8 = nn.Conv2d(76, 100, (2, 3))
        self.conv_4_9 = nn.Conv2d(100, 126, (2, 3))
        self.conv_4_10 = nn.Conv2d(126, 150, (2, 3))

        self.conv_4_11 = nn.Conv2d(150, 176, (2, 3))
        self.conv_4_12 = nn.Conv2d(176, 200, (2, 3))
        self.conv_4_13 = nn.Conv2d(200, 236, (2, 3))
        self.conv_4_14 = nn.Conv2d(236, 260, (2, 3))

        self.fc_4_1 = nn.Linear(260 * 6 * 9, 2500)
        self.fc_4_2 = nn.Linear(2500, 450)
        self.fc_4_3 = nn.Linear(450, 60)

    def predict_img_6(self, x):
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.relu(self.conv_4_4(x))
        x = F.relu(self.conv_4_5(x))
        x = self.pool(F.relu(self.conv_4_6(x)))

        x = F.relu(self.conv_4_7(x))
        x = F.relu(self.conv_4_8(x))
        x = F.relu(self.conv_4_9(x))
        x = self.pool(F.relu(self.conv_4_10(x)))

        x = F.relu(self.conv_4_11(x))
        x = F.relu(self.conv_4_12(x))
        x = F.relu(self.conv_4_13(x))
        x = F.relu(self.conv_4_14(x))

        x = x.view(-1, 260 * 6 * 9)

        x = F.relu(self.fc_4_1(x))
        x = F.relu(self.fc_4_2(x))
        x = self.fc_4_3(x)
        return x
    
    def forward(self, X: torch.Tensor):
        if self.single_frame:
            n = int(len(X) / 2)
            res = X[n].unsqueeze(0)
            return self.predict_img_6(res)

        predicted = []
        for img in X:
            k = self.predict_img_6(img.unsqueeze(0))
            predicted.append(k)
        stacked = torch.stack(predicted)
        mn = torch.mean(stacked, dim=0)
        return mn
