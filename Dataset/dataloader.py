import cv2
import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob


cfg = yaml.safe_load(open('config.yml', 'r'))
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")

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

def show_video(sample, name='Test'):
    if isinstance(sample, dict):
        for f in sample['frames']:
            cv2.imshow(sample['name'], f)
            cv2.waitKey(30)
    else:
        for f in sample:
            cv2.imshow(name, f)
            cv2.waitKey(30)

def get_meta(filename, as_int=True):
    A = filename[-11:-8]
    R = filename[-15:-12]
    P = filename[-19:-16]
    C = filename[-23:-20]
    S = filename[-27:-24]
    if as_int:
        return [int(A), int(R), int(P), int(C), int(S)]
    return [A, R, P, C, S]

class NTUDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = [f for f in files if get_meta(f)[0] <= cfg['max_actions']]
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
        video = torch.stack(video)
        sample = {'frames': video, 'class': int(filename[-11:-8]), 'name': filename}
        return sample

def test_print_generator(generator, max_batch=20):
    for i, item in enumerate(train_generator):
        X = item['frames'].numpy()
        y = item['class']
        names = item['name']
        print(f'batch: {i}, name: {names[0]}, frames: {len(X[0])}, height: {len(X[0][0])}, width: {len(X[0][0][0])}, class: {y[0]}')
        if i == max_batch:
            break

preprocessed_path = f'{cfg["preprocessed_path"]}/*/*'
raw_path = f'{cfg["dataset_path"]}/*/*'

#all_rgb = glob.glob(raw_path)
all_rgb = glob.glob(preprocessed_path)

partition = {}
partition['train'] = [f for f in all_rgb if get_meta(f)[4] in cfg["train_subjects"]]
partition['test'] = [f for f in all_rgb if get_meta(f)[4] not in cfg["train_subjects"]]

tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4, 0.4, 0.4), (0.4, 0.4, 0.4))
])

test_dataset = NTUDataset(partition['test'], tr)
train_dataset = NTUDataset(partition['train'], tr)

train_generator = DataLoader(train_dataset, **cfg['dataloader_params'])
test_generator = DataLoader(test_dataset, **cfg['dataloader_params'])
