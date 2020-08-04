import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class UnitConv(nn.Module):
    def __init__(self, D_in, D_out, kernel_size=(3, 3, 3), stride=1, dropout=0):

        super(UnitConv, self).__init__()
        D, H, W = (kernel_size, kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        padding = (int((D - 1) / 2), int((H - 1) / 2), int((W - 1) / 2))
        sd, sh, sw = (stride, stride, stride) if type(stride) == int else stride
        self.bn = nn.BatchNorm3d(D_out)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv3d(
            D_in,
            D_out,
            kernel_size = (D, H, W),
            padding = padding,
            stride = (sd, sh, sw),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.drop(x)
        return x

class UnitTCN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=9, stride=1):
        super(UnitTCN, self).__init__()
        self.unit1_1 = UnitConv(
            in_channel,
            out_channel,
            kernel_size = kernel_size,
            dropout = 0.0,
            stride = stride,
        )

        if in_channel != out_channel:
            self.down1 = UnitConv(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.down1 = None

    def forward(self, x):
        x = self.unit1_1(x) + (x if self.down1 is None else self.down1(x))
        return x

class BestTCNModelConv3d(nn.Module):
        def __init__(self, device):
            super(BestTCNModelConv3d, self).__init__()
            self.name = 'BestTCN'
            self.device = device
            self.input_size = 20 * 108 * 192
            self.batch_dim = 10

            self.unit0 = UnitTCN(3, 6, kernel_size=(3, 5, 7), stride=1)
            self.bn0 = nn.BatchNorm3d(6)

            self.unit1 = UnitTCN(6, 10, kernel_size=(3, 5, 7), stride=1)
            self.unit1_1 = UnitTCN(10, 10, kernel_size=(3, 5, 7), stride=1)
            self.bn1 = nn.BatchNorm3d(10)

            self.unit2 = UnitTCN(10, 16, kernel_size=(3, 5, 7), stride=1)
            self.unit2_1 = UnitTCN(16, 16, kernel_size=(3, 5, 7), stride=1)
            self.bn2 = nn.BatchNorm3d(16)

            self.unit3 = UnitTCN(16, 24, kernel_size=(3, 5, 5), stride=1)
            self.unit3_1 = UnitTCN(24, 24, kernel_size=(5, 5, 5), stride=1)
            self.bn3 = nn.BatchNorm3d(24)

            self.unit4 = UnitTCN(24, 36, kernel_size=(5, 5, 5), stride=1)
            self.unit4_1 = UnitTCN(36, 36, kernel_size=(5, 5, 5), stride=1)
            self.bn4 = nn.BatchNorm3d(36)

            self.unit5 = UnitTCN(36, 48, kernel_size=(5, 3, 5), stride=1)
            self.unit5_1 = UnitTCN(48, 48, kernel_size=(5, 3, 5), stride=1)
            self.bn5 = nn.BatchNorm3d(48)

            self.unit6 = UnitTCN(48, 72, kernel_size=(3, 3, 3), stride=1)
            self.unit6_1 = UnitTCN(72, 72, kernel_size=(3, 3, 3), stride=1)
            self.bn6 = nn.BatchNorm3d(72)

            self.relu = F.relu
            self.fc1 = nn.Linear(6480, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 60)

        def predict_4(self, x):
            x = self.unit0(x)
            x = F.avg_pool3d(self.unit1(x), kernel_size=(1, 2, 2))
            x = self.unit1_1(x)
            x = F.avg_pool3d(self.unit2(x), kernel_size=(1, 2, 2))
            x = self.unit2_1(x)
            x = F.avg_pool3d(self.unit3(x), kernel_size=(1, 2, 2))
            x = self.unit3_1(x)
            x = F.avg_pool3d(self.unit4(x), kernel_size=(1, 2, 2))
            x = self.unit4_1(x)
            x = F.avg_pool3d(self.unit5(x), kernel_size=(2, 2, 2))
            x = self.unit5_1(x)
            x = F.avg_pool3d(self.unit6(x), kernel_size=(2, 1, 1))
            x = self.unit6_1(x)

            x = x.view(10, -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def forward(self, x):
            x = x.transpose(1, 2)
            return self.predict_4(x)