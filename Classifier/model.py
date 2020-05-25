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

        # predict_rgb_sl_0
        # self.conv1 = nn.Conv3d(3, 6, (3, 5, 5))
        # self.bn1 = nn.BatchNorm3d(6)
        # self.conv2 = nn.Conv3d(6, 16, (3, 7, 3))
        # self.bn2 = nn.BatchNorm3d(16)
        # self.conv3 = nn.Conv3d(16, 32, (5, 7, 3))
        # self.bn3 = nn.BatchNorm3d(32)
        # self.fc1 = nn.Linear(32 * 8 * 15, 1024)
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 60)

        # predict_rgb_sl_2_02 4 x 5 x 10 / 2 x 3 x 6
        # self.conv1 = nn.Conv3d(3, 6, (3, 5, 7), padding=(1, 1, 3))
        # self.bn1 = nn.BatchNorm3d(6)
        # self.conv2 = nn.Conv3d(6, 12, (3, 3, 5), padding=(1, 0, 2))
        # self.bn2 = nn.BatchNorm3d(12)
        # self.conv3 = nn.Conv3d(12, 24, (3, 3, 3), padding=(0, 1, 1))
        # #self.conv3 = nn.Conv3d(12, 24, (3, 3, 3), padding=(1, 1, 1)) # _03
        # self.bn3 = nn.BatchNorm3d(24)
        # self.conv4 = nn.Conv3d(24, 48, (5, 2, 3), padding=(0, 0, 0))
        # #self.conv4 = nn.Conv3d(24, 48, (5, 2, 3), padding=(0, 0, 0)) # _03
        # self.bn4 = nn.BatchNorm3d(48)
        # self.conv5 = nn.Conv3d(48, 96, (3, 3, 5))
        # #self.conv5 = nn.Conv3d(48, 96, (3, 3, 5)) # _03
        # self.bn5 = nn.BatchNorm3d(96)
        # self.fc1 = nn.Linear(36 * 96, 1024)
        # #self.fc1 = nn.Linear(72 * 96, 1024) # _03
        # self.fc2 = nn.Linear(1024, 256)
        # self.fc3 = nn.Linear(256, 60)

        # lstm_0
        # self.input_size = 3 * 54 * 96
        # self.hidden_size = 128
        # self.output_size = 60
        # self.nr_hidden = 10
        # self.batch_dim = cfg['dataloader_params']['batch_size']
        # self.nr_frames = cfg['max_frames']
        # self.i2h = torch.randn(self.nr_hidden, self.batch_dim, self.hidden_size).to(device)
        # self.i2o = torch.randn(self.nr_hidden, self.batch_dim, self.hidden_size).to(device)
        # self.hidden = (self.i2h, self.i2o)
        # self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.nr_hidden, batch_first=True)
        # self.fc1 = nn.Linear(1280, 256)
        # self.fc2 = nn.Linear(256, 60)

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

    def predict_rgb_sl_0(self, x):
        x = self.pool2(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 32 * 8 * 15)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_lstm_0(self, x: torch.Tensor):
        x = x.reshape(self.batch_dim, self.nr_frames, self.input_size)
        x, hidden = self.lstm(x, self.hidden)
        x = x.reshape(self.batch_dim, 1280)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict_rgb_sl_2(self, x: torch.Tensor):
        x = self.pool2(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, 96 * 36)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward(self, x):
        #x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        return self.predict_rgb_d2_sl_00(x)