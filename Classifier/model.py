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

        self.conv1 = nn.Conv3d(3, 6, (3, 3, 5), padding=(1, 1, 2)) # 108x96
        self.conv1_1= nn.Conv3d(6, 6, (3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(6)

        self.conv2 = nn.Conv3d(6, 10, (3, 3, 3), padding=(1, 1, 1)) # 54x96
        self.conv2_1= nn.Conv3d(10, 10, (3, 3, 5), padding=(1, 1, 2))
        self.bn2 = nn.BatchNorm3d(10)

        self.conv3 = nn.Conv3d(10, 16, (3, 3, 5), padding=(1, 1, 2)) # 54x48
        self.conv3_1= nn.Conv3d(16, 16, (3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(16)

        self.conv4 = nn.Conv3d(16, 24, (3, 3, 3), padding=(1, 1, 1)) # 27x24
        self.conv4_1= nn.Conv3d(24, 24, (3, 3, 3), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(24)

        self.conv5 = nn.Conv3d(24, 32, (3, 3, 3), padding=(1, 1, 1)) # 13x24
        self.conv5_1= nn.Conv3d(32, 32, (3, 3, 5), padding=(1, 1, 2))
        self.bn5 = nn.BatchNorm3d(32)

        self.conv6 = nn.Conv3d(32, 40, (3, 3, 5), padding=(1, 1, 2)) # 13x12
        self.conv6_1= nn.Conv3d(40, 40, (3, 3, 3), padding=(1, 1, 1))
        self.bn6 = nn.BatchNorm3d(40)

        self.conv7 = nn.Conv3d(40, 64, (3, 3, 3), padding=(1, 1, 1)) # 6x12
        self.conv7_1= nn.Conv3d(64, 64, (3, 3, 3), padding=(1, 1, 1))
        self.bn7 = nn.BatchNorm3d(64)

        self.conv8 = nn.Conv3d(64, 96, (3, 3, 3), padding=(1, 1, 1)) # 6x6
        self.conv8_1= nn.Conv3d(96, 96, (3, 3, 3), padding=(1, 1, 1))
        self.bn8 = nn.BatchNorm3d(96)

        self.fcdrop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6912, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 60)

    def predict_rgb_d2_sl_00(self, x):
        x = F.avg_pool3d(F.leaky_relu(self.bn1(self.conv1(x))), kernel_size=(1, 1, 2)) # 20x108x96
        x = F.leaky_relu(self.bn1(self.conv1_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn2(self.conv2(x))), kernel_size=(1, 2, 1)) # 20x54x96
        x = F.leaky_relu(self.bn2(self.conv2_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn3(self.conv3(x))), kernel_size=(1, 1, 2)) # 20x54x48
        x = F.leaky_relu(self.bn3(self.conv3_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn4(self.conv4(x))), kernel_size=(1, 2, 2)) # 20x27x24
        x = F.leaky_relu(self.bn4(self.conv4_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn5(self.conv5(x))), kernel_size=(2, 2, 1)) # 10x13x24
        x = F.leaky_relu(self.bn5(self.conv5_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn6(self.conv6(x))), kernel_size=(1, 1, 2)) # 10x13x12
        x = F.leaky_relu(self.bn6(self.conv6_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn7(self.conv7(x))), kernel_size=(2, 2, 1)) # 5x6x12
        x = F.leaky_relu(self.bn7(self.conv7_1(x)))

        x = F.avg_pool3d(F.leaky_relu(self.bn8(self.conv8(x))), kernel_size=(2, 1, 2)) # 2x6x6
        x = F.leaky_relu(self.bn8(self.conv8_1(x)))

        x = x.view(10, -1)
        x = self.fcdrop(F.relu(self.fc1(x)))
        x = self.fcdrop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.predict_rgb_d2_sl_00(x)