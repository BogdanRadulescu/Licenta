import torch
import cv2
import glob
import yaml
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Classifier.besttcn3 import BestTCNModelConv3d3

cfg = yaml.safe_load(open('config.yml', 'r'))
all_rgb = glob.glob(cfg["demo_path"])
t = cfg['actions']
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def video_to_rgb_array(name: str):
    capture = cv2.VideoCapture(name)
    capture.open(name)
    images = []
    while True:
        ret, img = capture.read()
        if ret:
            images.append(img)
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    return images

class NTUDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.files[idx]
        video = video_to_rgb_array(filename)
        if self.transform:
            video = [self.transform(x) for x in video]
        video = torch.stack(video).to(device)
        sample = {'frames': video, 'class': int(filename[-11:-8]), 'name': filename}
        return sample

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = NTUDataset(all_rgb, tr)
generator = DataLoader(dataset, **cfg['dataloader_params'])

model = BestTCNModelConv3d3(device)
model.load_state_dict(torch.load(f'{cfg["model_path"]}/{model.name}_00.pth'))
model.eval()
model.cuda(device)

with torch.no_grad():
    for items in generator:
        X = items['frames']
        y = int(items['class'][0].data)
        outputs = model(X)
        predicted = int([torch.argmax(x) for x in outputs.data][0].data) + 1
        print(items['name'][0])
        print(f'"{t[y]}" was predicted as "{t[predicted]}"!\n')
