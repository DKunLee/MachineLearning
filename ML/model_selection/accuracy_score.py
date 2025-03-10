import numpy as np

def accuracy_score(y:np.ndarray, pred_y:np.ndarray):
    correct = 0
    for actual, pred in zip(y, pred_y):
        if actual == pred:
            correct += 1
    
    return correct / y.size

