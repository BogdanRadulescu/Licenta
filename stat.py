import yaml

cfg = yaml.safe_load(open('config.yml', 'r'))
C = cfg['max_actions']
#model_name = 'Baseline'
#model_name = 'NTU_RGB_Classifier'
model_name = 'besttnc3'
filename = model_name + '_matrix' + '_cv'
filename += '_reduced'

actions = [i+1 for i in range(60) if i+1 not in cfg['illegal_actions']]
# actions.index(y[i]) == y_pred[i]
# [actions.index(x) for x in items['class']]

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
        if total == 0:
            precision[i] = -1
        else:
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
        if total == 0:
            recall[i] = -1
        else:
            recall[i] /= total
    return recall

def restore_matrix(filename):
    file = open(filename, 'r')
    matrix = [[int(x) for x in line.split()] for line in file]
    file.close()
    return matrix

def compute_top(results, k, reverse):
    arr = [(i+1, results[i]) for i in range(len(results))]
    arr.sort(reverse=reverse, key=lambda x: x[1])
    i = 0
    while arr[i][1] == -1:
        i += 1
    return arr[i:i+k]

def checksum(matrix):
    total = 0
    for x in matrix:
        total += sum(x)
    return total

matrix = restore_matrix(filename)
print(f'Checksum {checksum(matrix)}')
print(compute_accuracy(matrix))
precision = compute_precision(matrix)
recall = compute_recall(matrix)
print('Precision')
print(compute_top(precision, 5, False))
print(compute_top(precision, 5, True))
print('Recall')
print(compute_top(recall, 5, False))
print(compute_top(recall, 5, True))