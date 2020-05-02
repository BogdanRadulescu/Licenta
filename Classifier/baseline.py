import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor, from_numpy, FloatTensor
import torch
import numpy as np
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class NNModelBase(nn.Module):
    def __init__(self, single_frame=True):
        super(NNModelBase, self).__init__()
        self.name = 'Baseline'
        self.single_frame = single_frame


        self.conv1 = nn.Conv2d(3, 6, (3, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, (3, 7))
        self.conv3 = nn.Conv2d(16, 32, (2, 3))
        self.fc1 = nn.Linear(32 * 11 * 18, 1024)
        self.fc2 = nn.Linear(1024, 200)
        self.fc3 = nn.Linear(200, cfg['max_actions'])

        self.conv_1 = nn.Conv2d(3, 6, 5)
        self.conv_2 = nn.Conv2d(6, 16, 7)
        self.conv_3 = nn.Conv2d(16, 36, 7)
        self.conv_4 = nn.Conv2d(36, 56, 3)
        self.conv_5 = nn.Conv2d(36, 56, 7)
        self.conv_6 = nn.Conv2d(56, 76, 7)
        self.conv_7 = nn.Conv2d(76, 96, (5, 3))
        self.fc_1 = nn.Linear(96 * 16 * 9, 3400)
        self.fc_2 = nn.Linear(3400, 800)
        self.fc_3 = nn.Linear(800, 250)
        self.fc_4 = nn.Linear(250, cfg['max_actions'])


        self.conv_2_1 = nn.Conv2d(3, 6, (3, 5))
        self.conv_2_2 = nn.Conv2d(6, 16, (3, 7))
        self.conv_2_3 = nn.Conv2d(16, 36, (3, 5))
        self.conv_2_4 = nn.Conv2d(36, 75, (3, 5))
        self.conv_2_5 = nn.Conv2d(75, 126, (3, 5))
        self.conv_2_6 = nn.Conv2d(126, 200, (3, 5))
        self.fc_2_1 = nn.Linear(200 * 16, 1500)
        self.fc_2_2 = nn.Linear(1500, 600)
        self.fc_2_3 = nn.Linear(600, 225)
        self.fc_2_4 = nn.Linear(225, 50)
        self.fc_2_5 = nn.Linear(50, cfg['max_actions'])


        self.conv_3_1 = nn.Conv2d(3, 6, (3, 3))
        self.conv_3_2 = nn.Conv2d(6, 16, (3, 3))
        self.conv_3_3 = nn.Conv2d(16, 26, (2, 3))
        self.conv_3_4 = nn.Conv2d(26, 36, (3, 3))
        self.conv_3_5 = nn.Conv2d(36, 46, (2, 2))
        self.conv_3_6 = nn.Conv2d(46, 56, (3, 3))
        self.conv_3_7 = nn.Conv2d(53, 66, (2, 3))
        self.conv_3_8 = nn.Conv2d(66, 76, (2, 3))
        self.conv_3_9 = nn.Conv2d(76, 86, (2, 3))
        self.fc_3_1 = nn.Linear(86 * 4 * 12, 1500)
        self.fc_3_2 = nn.Linear(1500, 550)
        self.fc_3_3 = nn.Linear(550, 200)
        self.fc_3_4 = nn.Linear(200, 70)
        self.fc_3_5 = nn.Linear(70, 10)

    def predict_img_4(self, X):
        x = F.relu(self.conv_3_1(x))
        x = self.pool(F.relu(self.conv_3_2(x)))
        x = F.relu(self.conv_3_3(x))
        x = self.pool(F.relu(self.conv_3_4(x)))
        x = F.relu(self.conv_3_5(x))
        x = F.relu(self.conv_3_6(x))
        x = F.relu(self.conv_3_7(x))
        x = F.relu(self.conv_3_8(x))
        x = F.relu(self.conv_3_9(x))
        x = x.view(-1, 86 * 4 * 12)
        x = F.relu(self.fc_3_1(x))
        x = F.relu(self.fc_3_2(x))
        x = F.relu(self.fc_3_3(x))
        x = F.relu(self.fc_3_4(x))
        x = self.fc_3_5(x)
        return x

    
    def predict_img_3(self, x):
        x = self.pool(F.relu(self.conv_2_1(x)))
        x = self.pool(F.relu(self.conv_2_2(x)))
        x = F.relu(self.conv_2_3(x))
        x = F.relu(self.conv_2_4(x))
        x = F.relu(self.conv_2_5(x))
        x = F.relu(self.conv_2_6(x))
        x = x.view(-1, 200 * 4 * 4)
        x = F.relu(self.fc_2_1(x))
        x = F.relu(self.fc_2_2(x))
        x = F.relu(self.fc_2_3(x))
        x = F.relu(self.fc_2_4(x))
        x = self.fc_2_5(x)
        return x.float()
    
    def predict_img(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 11 * 18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_img_4(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        x = self.pool(F.relu(self.conv_4(x)))
        x = self.pool(F.relu(self.conv_5(x)))
        x = self.pool(F.relu(self.conv_6(x)))
        x = F.relu(self.conv_7(x))
        x = x.view(-1, 96 * 16 * 9)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = self.fc_4(x)
        return x
    
    def forward(self, X):
        if self.single_frame:
            n = int(len(X) / 2)
            res = FloatTensor(np.reshape([X[n]], (1, 3, 54, 96)))
            return self.predict_img_3(res)

        predicted = []
        for img in X:
            k = self.predict_img(FloatTensor(np.reshape(img, (1, 3, 54, 96))))
            predicted.append(k)
        stacked = torch.stack(predicted)
        mn = torch.mean(stacked, dim=0)
        return mn
