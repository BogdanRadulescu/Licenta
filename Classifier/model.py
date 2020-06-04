import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class NNModel(nn.Module):
    def __init__(self, device):
        super(NNModel, self).__init__()
        self.name = 'NTU_RGB_Classifier'
        self.device = device
        self.pool = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
        self.nr_layers = 2

        # 184/ 2=> 92 - 8 = 84 / 2 => 42 - 6 = 36 / 2 = 18
        # predict_rgb_d2_sl_00 20 x 108 x 192 -> 4 x 4 x 6
        self.conv1 = nn.Conv3d(3, 6, (3, 5, 9), padding=(1, 0, 0))
        self.bn1 = nn.BatchNorm3d(6)
        self.conv2 = nn.Conv3d(6, 12, (3, 5, 9), padding=(1, 0, 0))
        self.bn2 = nn.BatchNorm3d(12)
        self.conv3 = nn.Conv3d(12, 24, (5, 5, 7), padding=(0, 0, 0))
        self.bn3 = nn.BatchNorm3d(24)
        # no pool
        self.conv4 = nn.Conv3d(24, 48, (5, 3, 5), padding=(0, 0, 0))
        self.bn4 = nn.BatchNorm3d(48)
        self.conv5 = nn.Conv3d(48, 72, (5, 3, 5), padding=(0, 0, 0))
        self.bn5 = nn.BatchNorm3d(72)
        self.conv6 = nn.Conv3d(72, 96, (5, 3, 5), padding=(0, 0, 0))
        self.bn6 = nn.BatchNorm3d(96)
        self.fc1 = nn.Linear(96 * 96, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 60)

    def predict_rgb_d2_sl_00(self, x):
        x = self.pool2(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = x.view(-1, 96 * 96)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward(self, x):
        #x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        return self.predict_rgb_d2_sl_00(x)