from Dataset.dataloader import device
from Classifier.baseline import *
from Classifier.model import *
from Classifier.besttcn3 import BestTCNModelConv3d3
import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))
# model = NNModelBase(device)
# model = NNModel(device)
model = BestTCNModelConv3d3(device)
filename = f'{model.name}_matrix_cv'
C = cfg['max_actions']

# for testing purposes
#filename = 'test'
#C = 3
#matrix = [[i * i + j + 1 for i in range(C)] for j in range(C)]

def print_matrix(matrix):
    for i in range(C):
        line = ''
        for j in range(C):
            line += f'{matrix[i][j]} '
        print(line)

def compute_accuracy(matrix):
    correct, total = 0.0, 0.0
    for i in range(C):
        for j in range(C):
            if i == j:
                correct += matrix[i][j]
            total += matrix[i][j]
    return 100 * correct / total

def compute_recall_sexy(matrix):
    return [matrix[i][i] / sum([matrix[j][i] for j in range(C)]) for i in range(C)]

def compute_precision_sexy(matrix):
    return [matrix[i][i] / sum([matrix[i][j] for j in range(C)]) for i in range(C)]

def compute_precision(matrix):
    precision = [0.0 for i in range(C)]
    for i in range(C):
        total = 0.0
        for j in range(C):
            if i == j:
                precision[i] += matrix[i][j]
            total += matrix[i][j]
        precision[i] /= total
    return precision

def compute_recall(matrix):
    recall = [0.0 for i in range(C)]
    for i in range(C):
        total = 0.0
        for j in range(C):
            if i == j:
                recall[i] += matrix[i][j]
            total += matrix[j][i]
        recall[i] /= total
    return recall

def restore_matrix(filename):
    file = open(filename, 'r')
    matrix = [[int(x) for x in line.split()] for line in file]
    file.close()
    return matrix

def compute_top(results, k, reverse):
    arr = [(i, results[i]) for i in range(len(results))]
    arr.sort(reverse=reverse, key=lambda x: x[1])
    return arr[:k]

matrix = restore_matrix(filename)
print(compute_accuracy(matrix))
precision = compute_precision(matrix)
recall = compute_recall(matrix)
print('Precision')
print(compute_top(precision, 5, False))
print(compute_top(precision, 5, True))
print('Recall')
print(compute_top(recall, 5, False))
print(compute_top(recall, 5, True))