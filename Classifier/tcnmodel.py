import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class UnitConv(nn.Module):
    def __init__(self, D_in, D_out, kernel_size=3, stride=1, dropout=0):

        super(UnitConv, self).__init__()
        self.bn = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            D_in,
            D_out,
            kernel_size = kernel_size,
            padding = int((kernel_size - 1) / 2),
            stride = stride,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv(x)
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

class TCNModelConv1d(nn.Module):
        def __init__(self, device):
            super(TCNModelConv1d, self).__init__()
            self.name = 'TCNModelConv1d'
            self.device = device
            self.input_size = 20 * 108 * 192
            self.batch_dim = 10

            #self.conv0 = nn.Conv1d(24, 48, kernel_size=13, padding=6)
            #self.conv1 = nn.Conv1d(48, 96, kernel_size=9, padding=4)
            self.unit1 = UnitTCN(3, 6, stride=2, kernel_size=9)
            self.bn1 = nn.BatchNorm1d(6)

            self.unit2 = UnitTCN(6, 12, stride=2, kernel_size=9)
            self.bn2 = nn.BatchNorm1d(12)

            self.unit3 = UnitTCN(12, 24, stride=2, kernel_size=9)
            self.bn3 = nn.BatchNorm1d(24)

            self.unit4 = UnitTCN(24, 48, stride=2, kernel_size=7)
            self.bn4 = nn.BatchNorm1d(48)

            self.unit5 = UnitTCN(48, 96, stride=2, kernel_size=7)
            self.bn5 = nn.BatchNorm1d(96)

            self.unit6 = UnitTCN(96, 192, stride=2, kernel_size=5)
            self.bn6 = nn.BatchNorm1d(192)

            self.unit7 = UnitTCN(192, 256, stride=2, kernel_size=5)
            self.bn7 = nn.BatchNorm1d(256)

            self.unit8 = UnitTCN(256, 512, stride=2, kernel_size=3)
            self.bn8 = nn.BatchNorm1d(512)

            self.relu = F.relu
            self.fc1 = nn.Linear(6656, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 60)
        
        def forward(self, x):
            x = x.reshape(self.batch_dim, 3, self.input_size)

            x = self.bn1(self.unit1(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn2(self.unit2(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn3(self.unit3(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn4(self.unit4(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn5(self.unit5(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn6(self.unit6(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn7(self.unit7(x))
            x = F.avg_pool1d(x, kernel_size=2)

            x = self.bn8(self.unit8(x))

            x = x.view(10, -1)
            x = self.relu(x)

            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x