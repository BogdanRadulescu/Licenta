import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class UnitConv(nn.Module):
    def __init__(self, D_in, D_out, d_in, kernel_size=(3, 3, 3), stride=1, dropout=0, padding=None):

        super(UnitConv, self).__init__()
        D, H, W = (kernel_size, kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
        padding = (int((D - 1) / 2), int((H - 1) / 2), int((W - 1) / 2))
        sd, sh, sw = (stride, stride, stride) if type(stride) == int else stride
        self.bn = nn.BatchNorm3d(D_out)
        #self.bn = nn.LayerNorm(d_in)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout3d(dropout)
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
        x = self.drop(x)
        return x

class UnitTCN(nn.Module):
    def __init__(self, in_channel, out_channel, d_in, kernel_size=9, stride=1, dropout=0.5):
        super(UnitTCN, self).__init__()
        self.unit1_1 = UnitConv(
            in_channel,
            out_channel,
            kernel_size = kernel_size,
            dropout = dropout,
            stride = stride,
            d_in=d_in,
        )

        if in_channel != out_channel:
            # kernel 1 -> version 01; kernel 3 -> version 00
            self.down1 = UnitConv(in_channel, out_channel, kernel_size=3, stride=stride, d_in=d_in)
        else:
            self.down1 = None

    def forward(self, x):
        x = self.unit1_1(x) + (x if self.down1 is None else self.down1(x))
        return x

class BestTCNModelConv3d3(nn.Module):
        def __init__(self, device):
            super(BestTCNModelConv3d3, self).__init__()
            self.name = 'besttnc3'
            self.device = device
            self.input_size = 20 * 108 * 192
            self.batch_dim = 10

            self.unit0 = UnitTCN(3, 3, kernel_size=(3, 3, 5), stride=1, dropout=0.2, d_in=(3, 20, 108, 192))

            self.unit1 = UnitTCN(3, 6, kernel_size=(3, 3, 5), stride=1, dropout=0.2, d_in=(6, 20, 108, 192))
            self.unit1_1 = UnitTCN(6, 6, kernel_size=(3, 3, 3), stride=1, dropout=0.2, d_in=(6, 20, 108, 96))

            self.unit2 = UnitTCN(6, 10, kernel_size=(3, 3, 3), stride=1, dropout=0.2, d_in=(10, 20, 108, 96))
            self.unit2_1 = UnitTCN(10, 10, kernel_size=(3, 3, 5), stride=1, dropout=0.2, d_in=(10, 20, 54, 96))

            self.unit3 = UnitTCN(10, 16, kernel_size=(3, 3, 5), stride=1, dropout=0.2, d_in=(16, 20, 54, 96))
            self.unit3_1 = UnitTCN(16, 16, kernel_size=(3, 3, 3), stride=1, dropout=0.2, d_in=(16, 20, 54, 48))

            self.unit4 = UnitTCN(16, 24, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(24, 20, 54, 48))
            self.unit4_1 = UnitTCN(24, 24, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(24, 20, 27, 24))

            self.unit5 = UnitTCN(24, 32, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(32, 20, 27, 24))
            self.unit5_1 = UnitTCN(32, 32, kernel_size=(3, 3, 5), stride=1, dropout=0.3, d_in=(32, 10, 13, 24))

            self.unit6 = UnitTCN(32, 40, kernel_size=(3, 3, 5), stride=1, dropout=0.3, d_in=(40, 10, 13, 24))
            self.unit6_1 = UnitTCN(40, 40, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(40, 10, 13, 12))

            self.unit7 = UnitTCN(40, 64, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(64, 10, 13, 12))
            self.unit7_1 = UnitTCN(64, 64, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(64, 5, 6, 12))

            self.unit8 = UnitTCN(64, 96, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(96, 5, 6, 12))
            self.unit8_1 = UnitTCN(96, 96, kernel_size=(3, 3, 3), stride=1, dropout=0.3, d_in=(96, 2, 6, 6))

            self.fcdrop = nn.Dropout(0.5)

            self.fc1 = nn.Linear(6912, 1024)
            self.fc2 = nn.Linear(1024, 256)
            #self.fc3 = nn.Linear(256, 60)
            self.fc3 = nn.Linear(256, 45)

        def predict(self, x):
            x = self.unit0(x)
            x = F.avg_pool3d(self.unit1(x), kernel_size=(1, 1, 2)) # 20x108x96
            x = self.unit1_1(x)

            x = F.avg_pool3d(self.unit2(x), kernel_size=(1, 2, 1)) # 20x54x96
            x = self.unit2_1(x)

            x = F.avg_pool3d(self.unit3(x), kernel_size=(1, 1, 2)) # 20x54x48
            x = self.unit3_1(x)

            x = F.avg_pool3d(self.unit4(x), kernel_size=(1, 2, 2)) # 20x27x24
            x = self.unit4_1(x)

            x = F.avg_pool3d(self.unit5(x), kernel_size=(2, 2, 1)) # 10x13x24
            x = self.unit5_1(x)

            x = F.avg_pool3d(self.unit6(x), kernel_size=(1, 1, 2)) # 10x13x12
            x = self.unit6_1(x)

            x = F.avg_pool3d(self.unit7(x), kernel_size=(2, 2, 1)) #5x6x12
            x = self.unit7_1(x)

            x = F.avg_pool3d(self.unit8(x), kernel_size=(2, 1, 2)) #2x6x6
            x = self.unit8_1(x)

            x = x.view(10, -1)
            x = self.fcdrop(F.leaky_relu(self.fc1(x)))
            x = self.fcdrop(F.leaky_relu(self.fc2(x)))
            x = self.fc3(x)
            return x

        def forward(self, x):
            x = x.transpose(1, 2)
            return self.predict(x)