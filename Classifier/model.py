import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))

class NNModel(nn.Module):
    def __init__(self):
        super(NNModel, self).__init__()
        self.name = 'NTU_RGB_Classifier'
        self.conv1 = nn.Conv3d(3, 6, (3, 5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv3d(6, 16, (3, 7, 3))
        self.conv3 = nn.Conv3d(16, 32, (5, 7, 3))
        self.fc1 = nn.Linear(32 * 8 * 14 * 2, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, cfg['max_actions'])
    
    def forward(self, x):
        x = [self.pool(k) for k in F.relu(self.conv1(x))]
        x = [self.pool(k) for k in F.relu(self.conv2(x))]
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * 8 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)