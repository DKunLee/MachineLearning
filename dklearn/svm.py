import numpy as np
import copy


class SVC():
    """
    Linear kernel SVM via SGD.

    Attributes:
        num_features (int): Dimensionality of input.
        max_iter (int): Number of epochs.
        lr0 (float): Initial learning rate.
        C (float): Regularization parameter.
        w (np.ndarray): Weight vector.
        b (float): Bias term.
    """
    def __init__(self, num_features:int, max_iter:int=100, lr0:float=0.01, C:float=2):
        """
        Initialize the SVM model.

        Args:
            num_features (int)
            max_iter (int)
            lr0 (float)
            C (float)
        """
        self.max_iter = max_iter
        self.lr0 = lr0
        self.lr = lr0
        self.C = C
        self.w = np.random.uniform(-0.01, 0.01, size=num_features)
        self.b = 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train SVM via stochastic subgradient descent.

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Binary labels (0 or 1).
        """
        y = np.where(y==0, -1, 1)
        t = 0

        for epoch in range(self.max_iter):
            shuffled_x, shuffled_y = self.shuffle_data(X, y)
            n = shuffled_x.shape[0]

            for i in range(n):
                x_i = shuffled_x[i]
                y_i = shuffled_y[i]

                if y_i*(np.matmul(self.w, x_i)) <= 1:
                    self.w = (1-self.lr)*self.w + self.lr*self.C*y_i*x_i
                    self.b = self.b + self.lr * self.C * y_i
                else:
                    self.w = (1-self.lr)*self.w

            t += 1
            self.lr = self.lr / (1+t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X (np.ndarray)

        Returns:
            np.ndarray: Predicted labels (0 or 1).
        """
        preds = np.matmul(X, self.w) + self.b
        return np.where(preds>=0, 1, 0)
    
    def get_params(self, deep=True):
        """
        Get estimator parameters.

        Args:
            deep (bool)

        Returns:
            dict
        """
        params = {
            'max_iter': self.max_iter,
            'lr0': self.lr0,
            'C': self.C,
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        """
        Set estimator parameters.

        Args:
            **params

        Returns:
            self

        Raises:
            ValueError
        """
        for key, value in params.items():
            if key not in self.get_params(deep=False):
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self

    # - Helper Method
    def shuffle_data(self, X: np.ndarray, y: np.ndarray):
        """
        Shuffle features and labels in tandem.
        """
        assert len(X) == len(y), f'{len(X)=} and {len(y)=} must have the same length in dimension 0'
        p = np.random.permutation(len(X))
        return X[p], y[p]
    