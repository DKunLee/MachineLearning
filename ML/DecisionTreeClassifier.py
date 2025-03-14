import numpy as np
from math import log2


class DecisionTreeClassifier():
    """
    A Decision Tree Classifier implemented using the ID3 algorithm with entropy-based splitting.
    Supports both categorical and numerical attributes.
    """
    class __Node:
        """
        A private class representing a node in the decision tree.
        """
        def __init__(self, attribute=None, threshold=None, label=None):
            """
            Initialize a tree node.
            
            Parameters:
            - attribute (int or None): The index of the feature to split on.
            - threshold (float or None): The threshold value for numeric splits.
            - label (int or None): The class label if it's a leaf node.
            """
            self.attrubute = attribute
            self.threshold = threshold
            self.label = label
            self.children = dict()

    
    def __init__(self, max_depth:int=None, criterion:str="entropy", algorithm:str="ID3"):
        """
        Initialize the Decision Tree Classifier.
        
        Parameters:
        - max_depth (int or None): The maximum depth of the tree (None means unlimited depth).
        - criterion (str): The metric used for split selection ('entropy' only).
        - algorithm (str): The algorithm type ('ID3' only, others not implemented).
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.algorithm = algorithm
        self.root = None
        self.majority_label = None

    # Done
    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Train the decision tree model using the given dataset.
        
        Parameters:
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (np.ndarray): Target labels of shape (n_samples,).
        """
        self.majority_label = self.__major_label(y)

        if self.criterion == "entropy":
            if self.algorithm == "ID3":
                attrs = list(np.arange(0, X.shape[1]))
                self.root = self.__ID3(X, y, attrs, self.max_depth)
            else:
                raise ValueError("No such algorithm or hasn't implemented yet.")
        else:
            raise ValueError("No such information gain criterion or hasn't implemented yet.")
    
    def __ID3(self, X:np.ndarray, y:np.ndarray, attrs:list, depth_limit:int=None) -> __Node:
        """
        Recursively build the decision tree using the ID3 algorithm.
        
        Parameters:
        - X (np.ndarray): Feature matrix.
        - y (np.ndarray): Target labels.
        - attrs (list): List of available attributes to split on.
        - depth_limit (int or None): Maximum allowed depth.
        
        Returns:
        - __Node: The root node of the constructed tree.
        """
        if depth_limit == 0 or len(attrs) == 0:
            return self.__Node(label=self.majority_label)
        if np.unique(y).size == 1:
            return self.__Node(label=y[0])
        
        best_attr = self.__best_ig(X, y, attrs)
        attrs.remove(best_attr)
        node = self.__Node(attribute=best_attr)

        x = X[:, best_attr]

        if self.__is_numeric(x):
            _, t = self.__best_t(x, y)
            node.threshold = t
            
            sub_i = x>t
            L_x, L_y = X[~sub_i], y[~sub_i]
            R_x, R_y = X[sub_i], y[sub_i]

            node.children[f"<={t}"] = self.__ID3(L_x, L_y, attrs, None if depth_limit is None else depth_limit-1)
            node.children[f">{t}"] = self.__ID3(R_x, R_y, attrs, None if depth_limit is None else depth_limit-1)
        else:
            values = np.unique(x)
            for val in values:
                sub_i = x==val
                sub_x, sub_y = x[sub_i], y[sub_i]
                
                node.children[val] = self.__ID3(sub_x, sub_y, attrs, None if depth_limit is None else depth_limit-1)

        return node

    
    def __best_ig(self, X:np.ndarray, y:np.ndarray, attrs:list):
        """
        Determine the best attribute to split on by calculating information gain.
        
        Parameters:
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (np.ndarray): Target labels of shape (n_samples,).
        - attrs (list): List of available attributes to evaluate.
        
        Returns:
        - int: Index of the best attribute to split on.
        """
        total_entropy = self.__entropy(y)
        total_len = y.size
        max_ig = float('-inf')
        best_attr = None

        for attr in attrs:
            x = X[:, attr]
            values, counts = np.unique(x, return_counts=True)

            exp_entropy = 0
            ig = 0

            if self.__is_numeric(x):
                ig, _ = self.__best_t(x, y)
            else:
                for val, cnt in zip(values, counts):
                    sub_y = y[x==val]
                    exp_entropy += ((cnt/total_len)*self.__entropy(sub_y))
                ig = total_entropy - exp_entropy
            
            if ig > max_ig:
                max_ig = ig
                best_attr = attr
        
        return best_attr


    def __best_t(self, x:np.ndarray, y:np.ndarray):
        """
        Determine the best threshold for splitting a numerical attribute.
        
        Parameters:
        - x (np.ndarray): The numerical feature column.
        - y (np.ndarray): The corresponding target labels.
        
        Returns:
        - tuple: (float, float) The best information gain and corresponding threshold.
        """
        t = None
        total_entropy = self.__entropy(y)
        total_len = y.size
        values = np.unique(x)
        max_ig = float('-inf')

        sp_points = [(values[i-1]+values[i])/2 for i in range(1, values.size)]

        for temp_t in sp_points:
            greater_than_temp_t = x>temp_t 
            L_y, R_y = y[~greater_than_temp_t], y[greater_than_temp_t]

            exp_entropy = (L_y.size/total_len)*self.__entropy(L_y) + (R_y.size/total_len)*self.__entropy(R_y)
            ig = total_entropy - exp_entropy
            if ig > max_ig:
                max_ig = ig
                t = temp_t
        
        return max_ig, t

    def __entropy(self, y:np.ndarray):
        """
        Compute the entropy of a given set of labels.
        
        Parameters:
        - y (np.ndarray): Array of target labels.
        
        Returns:
        - float: Entropy value.
        """
        total_len = y.size
        values, counts = np.unique(y, return_counts=True)
        entropy = 0
        if values.size != 1:
            for cnt in counts:
                p = cnt/total_len
                entropy -= p*log2(p)
        
        return entropy
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Parameters:
        - X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        
        Returns:
        - np.ndarray: Predicted class labels of shape (n_samples,).
        """
        rtn = []
        for r in X:
            rtn.append(self.__predictor(list(r), self.root))
        return np.array(rtn)
    
    def __predictor(self, row:list, node:__Node):
        """
        Recursively traverse the decision tree to make a prediction for a single instance.
        
        Parameters:
        - row (list): A single instance's feature values.
        - node (__Node): The current node in the decision tree.
        
        Returns:
        - int: Predicted label.
        """
        if node.label is not None:
            return node.label
        
        val = row[node.attrubute]

        if node.threshold is not None:
            chile_key = f">{node.threshold}" if float(val)>node.threshold else f"<={node.threshold}"
        else:
            chile_key = val

        if chile_key in node.children.keys():
            return self.__predictor(row, node.children[chile_key])
        else:
            return self.majority_label

    def __major_label(self, y:np.ndarray=None):
        """
        Find the most common class label in the dataset.
        
        Parameters:
        - y (np.ndarray): Array of target labels.
        
        Returns:
        - int: The majority class label.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    
    def __is_numeric(self, x:np.ndarray):
        """
        Check if a column contains numeric values.
        
        Parameters:
        - x (np.ndarray): A column of feature values.
        
        Returns:
        - bool: True if the column is numeric, otherwise False.
        """
        try:
            x.astype(float)
            return True
        except ValueError:
            return False
        
