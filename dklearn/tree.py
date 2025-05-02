import numpy as np
import pandas as pd
from typing import Union

from joblib import Parallel, delayed
import copy


class DecisionTreeClassifier():
    """
    Decision tree classifier using recursion and Gini impurity.
    """
    def __init__(self, criterion:str='gini', max_depth:int=None):
        """
        Args:
            criterion (str): Splitting criterion ('gini' only currently).
            max_depth (int, optional): Max tree depth.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.majority_label = None
    
    def fit(self, X:Union[np.ndarray, pd.DataFrame], y):
        """
        Build decision tree from training data.

        Args:
            X: Features (array or DataFrame).
            y: Labels.
        """
        self.majority_label = majority_label(y)
        attrs = None
        if isinstance(X, pd.DataFrame):
            attrs = X.columns.tolist()
        else:
            attrs = list(range(X.shape[1]))
        
        self.root = self.__build_tree(X, y, attrs, self.max_depth)

    def predict(self, X:Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels for samples.

        Returns:
            np.ndarray
        """
        predictions = [self.__predictor(row, self.root) for row in X]
        return np.array(predictions)
    
    def get_params(self, deep=True):
        """
        Get parameters of the tree.

        Returns:
            dict
        """
        params = {
            'max_depth': self.max_depth,
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        """
        Set tree parameters.

        Raises:
            ValueError
        """
        for key, value in params.items():
            if key not in self.get_params(deep=False):
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self

    class __Node():
        """
        Internal tree node.

        Attributes:
            attribute (int): Feature index.
            threshold (float): Split threshold.
            label: Class label for leaf nodes.
            children (dict): Map of condition→child node.
        """
        def __init__(self, attribute:int=None, threshold:float=None, label:str=None) -> None:
            self.attribute = attribute
            self.threshold = threshold
            self.label = label
            self.children = {}
        
    def __build_tree(self, X:Union[np.ndarray, pd.DataFrame], y:np.ndarray, attrs:list, depth:int=None):
        """
        Recursively build the decision tree.

        Args:
            X: Features (array or DataFrame).
            y: Labels.
            attrs: List of attributes to consider.
            depth: Current depth in the tree.

        Returns:
            __Node: The root node of the tree.
        """
        if len(np.unique(y))==1:
            return self.__Node(label=y[0])
        if not attrs or (depth is not None and depth<0):
            return self.__Node(label=majority_label(y))

        node = self.__Node()
        best_attr, t = self.__best_split(X, y, attrs)

        remaining_attrs = attrs.copy()
        remaining_attrs.remove(best_attr)

        node.attribute = best_attr
        node.threshold = t

        left_child = None
        right_child = None

        if isinstance(X, pd.DataFrame):
            left_mask = X[best_attr] <= t
            right_mask = X[best_attr] > t

            left_child = self.__build_tree(X[left_mask], y[left_mask], remaining_attrs, None if depth is None else depth-1)
            right_child = self.__build_tree(X[right_mask], y[right_mask], remaining_attrs, None if depth is None else depth-1)
        else:
            x = X[:, best_attr]

            left_mask = x<=t
            right_mask = x>t

            left_child = self.__build_tree(X[left_mask], y[left_mask], remaining_attrs, None if depth is None else depth-1)
            right_child = self.__build_tree(X[right_mask], y[right_mask], remaining_attrs, None if depth is None else depth-1)
        
        if (not left_child.children and not right_child.children and left_child.label==right_child.label):
            return self.__Node(label=left_child.label)
            
        node.children[f"<={t}"] = left_child
        node.children[f">{t}"] = right_child

        return node
    
    def __best_split(self, X:Union[np.ndarray, pd.DataFrame], y:np.ndarray, attrs:list):
        """
        Find the best attribute and threshold to split on.

        Args:
            X: Features (array or DataFrame).
            y: Labels.
            attrs: List of attributes to consider.

        Returns:
            tuple: (best attribute, best threshold)
        """
        best_attr = None
        best_threshold = None
        best_gini = float('inf')

        for attr in attrs:
            if isinstance(X, pd.DataFrame):
                x = X[attr]
            else:
                x = X[:, attr]

            thresholds = np.unique(x)
            for t in thresholds:
                left_mask = x<=t
                right_mask = x>t

                if len(y[left_mask])==0 or len(y[right_mask])==0:
                    continue

                L_gini = gini_impurity(y[left_mask])
                R_gini = gini_impurity(y[right_mask])

                L_weight = len(y[left_mask]) / len(y)
                R_weight = len(y[right_mask]) / len(y)

                weighted_gini = L_weight * L_gini + R_weight * R_gini

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_attr = attr
                    best_threshold = t

        return best_attr, best_threshold
    
    def __predictor(self, row:list, node:__Node):
        """
        Recursively predict the class label for a sample.

        Args:
            row: Sample data.
            node: Current node in the tree.

        Returns:
            The predicted class label.
        """
        if not node.children:
            return node.label
        
        attr = node.attribute
        val = row[attr]

        child_key = f">{node.threshold}" if val>node.threshold else f"<={node.threshold}"
        
        if child_key in node.children:
            return self.__predictor(row, node.children[child_key])
        else:
            return node.label
        
    def plot_tree(self):
        """
        Print the decision tree structure.

        Args:
            None

        Returns:
            Decision tree structure
        """
        def recurse(node, depth=0):
            indent = "  " * depth
            if not node.children:
                print(f"{indent}Predict: {node.label}")
                return

            for condition, child in node.children.items():
                print(f"{indent}If attribute[{node.attribute}] {condition}:")
                recurse(child, depth + 1)

        recurse(self.root)


def gini_impurity(y:np.ndarray):
    """
    Compute Gini impurity of a label array.

    Args:
        y (np.ndarray)

    Returns:
        float
    """
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / y.size
    return 1 - np.sum(probs**2)

def majority_label(y:np.ndarray):
    """
    Return the most frequent label in `y`.

    Args:
        y (np.ndarray)

    Returns:
        The label with highest count.
    """
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]