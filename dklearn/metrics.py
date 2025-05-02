import numpy as np

def accuracy_score(y:np.ndarray, pred_y:np.ndarray):
    """
    Compute the classification accuracy.

    Args:
        y (np.ndarray): True labels, shape (n_samples,).
        pred_y (np.ndarray): Predicted labels, same shape.

    Returns:
        float: Fraction of correct predictions.
    """
    correct = 0
    for actual, pred in zip(y, pred_y):
        if actual == pred:
            correct += 1
    
    return correct / y.size

def f1_score(y_true, y_pred):
    '''
    The f1 score evaluates the quality of a classifier using its precision p and recall r, where p = tp/(tp+fp) and
    r = tp/(tp+fn).

    The best value is 1 and the worst value is 0.

    The f1 score is f1 = 2*(p*r)/(p+r)

    Parameters:
    - y_true: The true labels of shape (n_samples,).
    - y_pred: The predicted labels of shape (n_samples,).
    '''
    
    if not isinstance(y_true, list):
        try:
            y_true = list(y_true)
        except:
            raise TypeError("Cannot convert the y_true to list")
    if not isinstance(y_pred, list):
        try:
            y_pred = list(y_pred)
        except:
            raise TypeError("Cannot convert the y_pred to list")

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)

    score = 0 if p+r==0 else 2*((p*r)/(p+r))

    return score

def precision_score(y_true, y_pred):
    '''
    The precision score is the ratio of true positives (tp) to all predicted positives (tp + fp) where 'tp' is the number of true
    positives and 'fp' the number of false positives.

    The best value is 1 and the worst value is 0.

    Parameters:
    - y_true: The true labels of shape (n_samples,).
    - y_pred: The predicted labels of shape (n_samples,). 
    '''
    if not isinstance(y_true, list):
        try:
            y_true = list(y_true)
        except:
            raise TypeError("Cannot convert the y_true to list")
    if not isinstance(y_pred, list):
        try:
            y_pred = list(y_pred)
        except:
            raise TypeError("Cannot convert the y_pred to list")

    tp = __true_positive_binary(y_true, y_pred)
    fp = __false_positive_binary(y_true, y_pred)
    
    score = 0 if tp+fp==0 else tp/(tp+fp)

    return score

def recall_score(y_true, y_pred, average:str='binary'):
    '''
    The recall score is the ratio of true positives (tp) to all actual positives (tp + fn) where 'tp' is the number of true
    positives and 'fn' the number of false negatives.

    The best value is 1 and the worst value is 0.

    Parameters:
    - y_true: The true labels of shape (n_samples,).
    - y_pred: The predicted labels of shape (n_samples,).
    '''
    if not isinstance(y_true, list):
        try:
            y_true = list(y_true)
        except:
            raise TypeError("Cannot convert the y_true to list")
    if not isinstance(y_pred, list):
        try:
            y_pred = list(y_pred)
        except:
            raise TypeError("Cannot convert the y_pred to list")

    tp = __true_positive_binary(y_true, y_pred)
    fn = __false_negative_binary(y_true, y_pred)
    
    score = 0 if tp+fn==0 else tp/(tp+fn)

    return score

    
def __true_positive_binary(y_true, y_pred):
    """Count true positives for binary classification."""
    count = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p > 0 and y_t==y_p:
            count += 1
    return count

def __false_positive_binary(y_true, y_pred):
    """Count false positives for binary classification."""
    count = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p > 0 and y_t!=y_p:
            count += 1
    return count

def __false_negative_binary(y_true, y_pred):
    """Count false negatives for binary classification."""
    count = 0
    for y_t, y_p in zip(y_true, y_pred):
        if y_p <= 0 and y_t!=y_p:
            count += 1
    return count

