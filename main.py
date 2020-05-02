from Dataset.dataloader import *
from Classifier.baseline import *
from Classifier.model import *
import glob
from time import sleep
import psutil
import torch.multiprocessing as mp
from subprocess import call
import torch.optim as optim
import torch.nn as nn
import os

criterion = nn.CrossEntropyLoss()
cfg = yaml.safe_load(open('config.yml', 'r'))

def load_model(ModelClass, path):
    model = ModelClass()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

def train_and_save_model(model, optimizer, path, display=True):
    global_running_loss = 0
    once_every = 10
    for epoch in range(cfg['max_epochs']):
        running_loss = 0
        for i, items in enumerate(train_generator):
            for j in range(len(items['frames'])):
                X = items['frames'][j].numpy()
                y = torch.tensor([items['class'][j]])
                name = items['name'][j]
                if not display:
                    print(f'batch: {i}, name: {name}, frames: {len(X)}, height: {len(X[0])}, width: {len(X[0][0])}, class: {y}')
                optimizer.zero_grad()
                y_pred = model(X)
                #print(y_pred)
                # y - 1 since classes start at 1
                loss = criterion(y_pred, y - 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if display and i % once_every == once_every - 1:
                print('[e %d, b %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / once_every))
                print(int(y - 1), int(torch.argmax(y_pred.data)))
                global_running_loss += running_loss
                running_loss = 0.0
            if i % (once_every * 10) == once_every * 10 - 1:
                torch.save(model.state_dict(), path)
    return global_running_loss
    

def load_and_test_model(ModelClass, path, display=True, once_every=20):
    model = ModelClass()
    model.load_state_dict(torch.load(path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, items in enumerate(test_generator):
            for j in range(len(items['frames'])):
                X = items['frames'][j].numpy()
                y = torch.tensor([items['class'][j]])
                name = items['name'][j]
                outputs = model(X)
                #print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                print(int(y), int(predicted))
                total += 1
                correct += (int(predicted) == int(y - 1))
            if i % once_every == once_every - 1:
                print(f'[{i + 1}]: Partial accuracy: {100 * correct / total}')
    if display:
        print('Accuracy: %d %%' % (100 * correct / total))
    return correct/total

def train_and_test_model(ModelClass, basePath, additional_name_path='', train=True, should_load_model=True):
    model = ModelClass()
    path = f'{basePath}\{model.name}{additional_name_path}.pth'
    if should_load_model:
        model = load_model(ModelClass, path)
    if train:
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        running_loss = train_and_save_model(model, optimizer, path)
    accuracy = load_and_test_model(ModelClass, path, True)
    return model.name, running_loss, accuracy

def main():
    
    name, running_loss, accuracy = None, None, None
    name, running_loss, accuracy = train_and_test_model(NNModelBase, cfg['base_model_path'], '_10_sf_02', train=True)
    #print(name, running_loss, accuracy)
    #name, running_loss, accuracy = train_and_test_model(NNModel, cfg['model_path'], '_00')
    #print(name, running_loss, accuracy)
    #test_print_generator(train_generator)


if __name__ == '__main__':
    main()