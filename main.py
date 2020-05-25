from Dataset.dataloader import *
from Classifier.baseline import *
from Classifier.model import *
from Classifier.tcnmodel2 import TCNModelConv3d
from Classifier.tcnmodel import TCNModelConv1d
from Classifier.besttcn import BestTCNModelConv3d
import glob
from time import sleep
import psutil
import torch.multiprocessing as mp
from subprocess import call
import torch.optim as optim
import torch.nn as nn
import os

criterion = nn.CrossEntropyLoss().cuda()
cfg = yaml.safe_load(open('config.yml', 'r'))

def load_model(ModelClass: torch.nn.Module, path) -> torch.nn.Module:
    model = ModelClass(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def train_and_save_model(model: torch.nn.Module, optimizer, path, display=True):
    global_running_loss = 0
    model.train()
    once_every = 50
    for epoch in range(cfg['max_epochs']):
        running_loss = 0
        for i, items in enumerate(train_generator):
            X = items['frames']#.to(device)
            y = [k - 1 for k in items['class']]
            y = torch.tensor(y).to(device)

            optimizer.zero_grad()
            try:
                y_pred = model(X)
            except:
                print('ERROR: Something occured, skipping data')
                names = [x[-28:] for x in items['name']]
                print(names)
                continue
            #print(y_pred.size(), X.size(), y.size())
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if display and i % once_every == once_every - 1:
                print('[e %d, b %5d] loss: %.3f, (p: %d, a: %d)' % (epoch + 1, i + 1, running_loss, int(torch.argmax(y_pred[0].data)), int(y[0])))
                global_running_loss += running_loss
                running_loss = 0.0
            if i % (once_every * 5) == once_every * 5 - 1:
                torch.save(model.state_dict(), path)
    return global_running_loss

def load_and_test_model(ModelClass: torch.nn.Module, path, display=True, once_every=20):
    model = ModelClass(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda(device)
    correct = 0
    total = 0
    once_every = 20
    with torch.no_grad():
        for i, items in enumerate(train_generator):
            X = items['frames']#.to(device)
            y = [k - 1 for k in items['class']]
            y = torch.tensor(y).to(device)
            outputs = model(X)
            #print(outputs)
            predicted = [torch.argmax(x) for x in outputs.data]
            #print(int(y), int(predicted))
            for j in range(len(y)):
                total += 1
                correct += (int(predicted[j]) == int(y[j]))
            if i % once_every == once_every - 1:
                print(f'[{i + 1}]: Partial accuracy: {100 * correct / total}')
    if display:
        print('Accuracy: %d %%' % (100 * correct / total))
    return correct/total

def train_and_test_model(ModelClass: torch.nn.Module, basePath, additional_name_path='', train=True, should_load_model=False):
    model = ModelClass(device)
    path = f'{basePath}\{model.name}{additional_name_path}.pth'
    if should_load_model:
        model = load_model(ModelClass, path)
    model.cuda(device)
    if train:
        optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
        #optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        running_loss = train_and_save_model(model, optimizer, path)
    accuracy = load_and_test_model(ModelClass, path, True)
    return model.name, running_loss, accuracy

def main():
    
    name, running_loss, accuracy = None, None, None
    #name, running_loss, accuracy = train_and_test_model(NNModelBase, cfg['base_model_path'], '_60_af_01', train=True, should_load_model=False)
    #print(name, running_loss, accuracy)
    #name, running_loss, accuracy = train_and_test_model(NNModel, cfg['model_path'], '_d2_sl_00', train=True, should_load_model=True)
    #train_and_test_model(TCNModelConv3d, cfg['model_path'], '_00', train=True, should_load_model=True)
    train_and_test_model(BestTCNModelConv3d, cfg['model_path'], '_00', train=True, should_load_model=True)
    #train_and_test_model(TCNModelConv1d, cfg['model_path'], '_00', train=True, should_load_model=True)
    #print(name, running_loss, accuracy)
    #test_print_generator(train_generator)


if __name__ == '__main__':
    main()