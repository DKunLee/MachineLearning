import numpy as np
import copy

from dklearn.base import shuffle_data

class Perceptron():
    """
    Simple binary Perceptron classifier.

    Attributes:
        fit_intercept (bool): Whether to learn a bias term.
        max_iter (int): Number of passes over the data.
        shuffle (bool): Whether to shuffle data each epoch.
        eta0 (float): Learning rate.
        random_state (int, optional): Seed for reproducibility.
        w (np.ndarray): Weight vector, shape (n_features,).
        b (float): Bias term.
        labels (np.ndarray): The two class labels.
    """
    def __init__(self, fit_intercept: bool=True, max_iter: int=1000, shuffle: bool=True, eta0: float=0.01, random_state: int=None):
        """
        Initialize a Perceptron.

        Args:
            fit_intercept (bool): Include bias term if True.
            max_iter (int): Maximum training epochs.
            shuffle (bool): Shuffle data each epoch.
            eta0 (float): Learning rate.
            random_state (int, optional): Seed for randomness.
        """
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.eta0 = eta0
        self.random_state = random_state
        self.labels = None

        self.w = None
        self.b = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Train the Perceptron model.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Training labels (binary), shape (n_samples,).

        Raises:
            ValueError: If not exactly two unique labels.
        """
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"Perceptron only supports binary classification, but got {len(labels)} labels.")
        if self.random_state!=None:
            np.random.seed(self.random_state)

        self.labels = labels

        y = np.where(y==0, -1, y)

        self.w = np.random.uniform(-0.01, 0.01, size=X.shape[1])
        self.b = np.random.uniform(-0.01, 0.01) if self.fit_intercept else 0

        for _ in range(self.max_iter):
            if self.shuffle:
                X_shuffled, y_shuffled = shuffle_data(X, y)
            else:
                X_shuffled, y_shuffled = X, y

            for i, (x_i, y_i) in enumerate(zip(X_shuffled, y_shuffled)):
                pred = np.dot(x_i, self.w) + (self.b if self.fit_intercept else 0)
                if y_i*pred<=0:
                    self.w += self.eta0 * y_i * x_i
                    if self.fit_intercept:
                        self.b += self.eta0 * y_i
        
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X (np.ndarray): Samples to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels, shape (n_samples,).

        Raises:
            ValueError: If called before `.fit()`.
        """
        if not self.w.any():
            raise ValueError("Estimator not fitted yet.")
        preds = []
        for x_i in X:
            pred = np.dot(x_i, self.w) + (self.b if self.fit_intercept else 0)
            preds.append(self.labels[1] if pred>=0 else self.labels[0])
        return np.array(preds)
    
    def get_params(self, deep=True):
        """
        Get estimator parameters.

        Args:
            deep (bool): If True, return copies of parameters.

        Returns:
            dict
        """
        params = {
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'shuffle': self.shuffle,
            'eta0': self.eta0,
            'random_state': self.random_state
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        """
        Set estimator parameters.

        Args:
            **params: Parameters to update.

        Returns:
            self

        Raises:
            ValueError: On invalid parameter names.
        """
        for key, value in params.items():
            if key not in self.get_params(deep=False):
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self
    
    
class LogisticRegression():
    """
    Binary logistic regression using SGD with L2‑regularization.

    Attributes:
        max_iter (int): Training epochs.
        eta0 (float): Base learning rate.
        sigma2 (float): Regularization strength.
        shuffle (bool): Shuffle data each epoch.
        random_state (int, optional): Seed for reproducibility.
        w (np.ndarray): Weight vector, shape (n_features,).
    """
    def __init__(self, max_iter: int=1000, eta0:float=0.1, sigma2:float=0.1, shuffle: bool=True, random_state: int=None):
        """
        Initialize logistic regression.

        Args:
            max_iter (int): Maximum number of epochs.
            eta0 (float): Initial learning rate.
            sigma2 (float): L2 regularization denominator.
            shuffle (bool): Shuffle data each epoch.
            random_state (int, optional): Random seed.
        """
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.eta0 = eta0
        self.sigma2 = sigma2
        self.random_state = random_state
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the logistic regression model via SGD.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Binary labels (0 or 1), shape (n_samples,).
        """
        if self.random_state!=None:
            np.random.seed(self.random_state)

        n = X.shape[1]
        self.w = np.random.uniform(-0.01, 0.01, size=n)

        y = np.where(y==0, -1, 1)
        t = 0

        lr = self.eta0

        for epoch in range(self.max_iter):
            if self.shuffle:
                shuffled_x, shuffled_y = self.shuffle_data(X, y)
            else:
                shuffled_x, shuffled_y = X, y

            n = shuffled_x.shape[0]

            for i in range(n):
                x_i = shuffled_x[i]
                y_i = shuffled_y[i]

                z = y_i * np.matmul(self.w, x_i)
                clipped_z = self.clip(z, max_abs_value=100)
                grad = -((y_i*x_i)/(1+np.exp(clipped_z))) + (2*self.w/self.sigma2)
                self.w = self.w - (lr*grad)

            t += 2
            lr = lr / (1+t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            List[int]: Predictions (0 or 1), shape (n_samples,).
        """
        preds = np.matmul(X, self.w)
        return [1 if self.sigmoid(p)>=0.5 else 0 for p in preds]
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool)

        Returns:
            dict
        """
        params = {
            'max_iter': self.max_iter,
            'eta0': self.eta0,
            'sigma2': self.sigma2,
            'shuffle': self.shuffle,
            'random_state': self.random_state
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        """
        Set parameters for this estimator.

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
        Shuffle features and labels (same as in base).
        """
        assert len(X) == len(y), f'{len(X)=} and {len(y)=} must have the same length in dimension 0'
        p = np.random.permutation(len(X))
        return X[p], y[p]
    
    def clip(self, x: np.ndarray, max_abs_value: float = 10000) -> np.ndarray:
        """
        Clip values elementwise to ±max_abs_value.
        """
        return np.minimum(np.maximum(x, -abs(max_abs_value)), abs(max_abs_value))
    
    def sigmoid(self, z: float) -> float:
        """
        Numerically stable sigmoid activation.

        Args:
            z (float)

        Returns:
            float: sigmoid(z)
        """
        Z = self.clip(z, max_abs_value=100)
        return 1 / (1 + np.exp(-Z))
    
    