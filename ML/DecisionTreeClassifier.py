import numpy as np
from math import log2


class DecisionTreeClassifier():
    class __Node:
        def __init__(self, attribute=None, threshold=None, label=None):
            self.attribute = attribute
            self.threshold = threshold
            # Can have children or label
            self.label = label
            self.children = {}
    

    def __init__(self, criterion:str="entropy", algo:str="ID3", max_depth:int=None):
        self.criterion = criterion
        self.algo = algo
        self.max_depth = max_depth
        self.root = None
        self.majority_label = None
    

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.majority_label = self.__majority_label(y)

        if self.criterion == "entropy":
            if self.algo == "ID3":
                self.root = self.__ID3(X, y, list(np.arange(0, X.shape[1])))
            else:
                raise ValueError("No such algorithm or hasn't implemented yet.")
        else:
            raise ValueError("No such information gain criterion or hasn't implemented yet.")
        
    
    def predict(self, X:np.ndarray):
        rtn = []
        for r in X:
            pred_y = self.__predictor(list(r), self.root)
            rtn.append(pred_y)
        return np.array(rtn)
            

    def __predictor(self, row:list, node:__Node):
        if node.label is not None:
            return node.label
        if node.attribute is None:
            return self.majority_label
        
        val = row[node.attribute]

        if node.threshold is not None:
            child_key = f">{node.threshold}" if float(val) > node.threshold else f"<={node.threshold}"
        else:
            child_key = val
        
        return self.__predictor(row, node.children[child_key])

    

    def __ID3(self, X:np.ndarray, y:np.ndarray, attrs:list, depth_limit=None):
        if (depth_limit == 0 or len(attrs) == 0):
            return self.__Node(label=self.__majority_label(y))
        vals = np.unique(y)
        if vals.size == 1:
            return self.__Node(label=vals[0])
        
        attr = self.__ig_classifier(X, y, attrs)
        attrs.remove(attr)
        node = self.__Node(attribute=attr)

        only_A = X[:, attr]
        
        if self.__is_numeric(only_A):
            _, t = self.__best_t(only_A, y)
            node.threshold = t

            sub_i = only_A > t
            sub_L_x, sub_L_y = X[~sub_i], y[~sub_i]
            sub_R_x, sub_R_y = X[sub_i], y[sub_i]

            node.children[f"<={t}"] = self.__ID3(sub_L_x, sub_L_y, attrs, None if depth_limit is None else depth_limit-1)
            node.children[f">{t}"] = self.__ID3(sub_R_x, sub_R_y, attrs, None if depth_limit is None else depth_limit-1)
        else:
            unique_vals = np.unique(only_A)
            for val in unique_vals:
                sub_i = only_A==val
                sub_x, sub_y = X[sub_i], y[sub_i]

                if sub_y.size==0:
                    node.children[val] = self.__Node(label=self.majority_label)
                else:
                    node.children[val] = self.__ID3(sub_x, sub_y, attrs, None if depth_limit is None else depth_limit-1)

        return node
    

    def __ig_classifier(self, X:np.ndarray, y:np.ndarray, attrs:list):
        total_entropy = self.__entropy(y)
        total_length = y.size
        max_ig = float('-inf')
        best_attr = None

        for i in attrs:
            x = X[:, i]
            unique_x = np.unique(x)
            if unique_x.size == 1:
                best_attr, max_ig = i, 1
                continue

            exp_entropy = 0
            ig = 0

            if self.__is_numeric(x):
                ig, t = self.__best_t(x, y)
            else:
                for val in unique_x:
                    sub_y = y[x==val]
                    exp_entropy += ((sub_y.size/total_length)*self.__entropy(sub_y))
                ig = total_entropy - exp_entropy
            
            if ig > max_ig:
                best_attr, max_ig = i, ig
        
        return best_attr


    def __best_t(self, x:np.ndarray, y:np.ndarray):
        t = 0
        total_entropy = self.__entropy(y)
        total_length = y.size
        unique_x = np.unique(x)
        max_ig = float('-inf')

        if unique_x.size == 1:
            return unique_x[0]
        
        # Potential split points
        sp_points = [(unique_x[i-1]+unique_x[i])/2 for i in range(1, unique_x.size)]
        for pnt in sp_points:
            sub_i = x>pnt
            sub_L_y, sub_R_y = y[~sub_i], y[sub_i]
            exp_entropy = ((sub_L_y.size/total_length)*self.__entropy(sub_L_y)) + ((sub_R_y.size/total_length)*self.__entropy(sub_R_y))
            sub_ig = total_entropy - exp_entropy
            if sub_ig > max_ig:
                max_ig, t = sub_ig, pnt
        
        return max_ig, t


    def __entropy(self, label:np.ndarray):
        total_length = label.size
        values, val_counts = np.unique(label, return_counts=True)
        if (values.size == 1):
            return 0
        
        entropy = 0
        for count in val_counts:
            p = count/total_length
            entropy -= p*log2(p)

        return entropy


    def __majority_label(self, y:np.ndarray):
        values, counts = np.unique(y, return_counts=True)
        most_freq_label = values[np.argmax(counts)]
        return most_freq_label
    

    def __is_numeric(self, col:np.ndarray):
        try:
            col.astype(float)
            return True
        except ValueError:
            return False

