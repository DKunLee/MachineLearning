import numpy as np
import copy


class AdaBoostClassifier():
    """
    AdaBoost ensemble classifier using any binary capable base estimator.

    Attributes:
        estimator: A single model instance supporting `.fit` and `.predict`.
        n_estimators (int): Maximum number of boosting rounds.
        random_state (int, optional): Seed for reproducibility.
        estimators (List): Fitted base estimators.
        alphas (List): Corresponding estimator weights.
        labels (np.ndarray): The two unique class labels seen during fitting.
    """
    def __init__(self, estimator, n_estimators=50, random_state=None):
        """
        Initialize AdaBoost.

        Args:
            estimator: Base learner (must support fit/predict).
            n_estimators (int): Number of boosting rounds.
            random_state (int, optional): Seed for randomness.
        """
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.labels = None
        self.estimators = []
        self.alphas = []

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Fit the AdaBoost model.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features).
            y (np.ndarray): Training labels, shape (n_samples,).

        Raises:
            ValueError: If more than two unique labels are provided.
        """
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError(f"Perceptron only supports binary classification, but got {len(labels)} labels.")
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.labels = labels
        
        y = np.where(y==0, -1, y)

        n = X.shape[0]
        D = np.ones(n)/n

        for _ in range(self.n_estimators):
            estimator = copy.deepcopy(self.estimator)
            estimator.fit(X, y)
            y_pred = estimator.predict(X)
            y_pred = np.where(y_pred==0, -1, y_pred)
            error = np.sum(D * (y_pred != y))

            if error >= 0.5:
                print("Error is greater than 0.5, stopping training")
                break
            
            alpha = 0.5 * np.log((1 - error + 1e-10) / (error + 1e-10))

            D = D * np.exp(-alpha * y * y_pred)
            D = D / np.sum(D)

            self.estimators.append(estimator)
            self.alphas.append(alpha)

    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        Predict labels for new data.

        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels, shape (n_samples,).

        Raises:
            ValueError: If called before `.fit()`.
        """
        if not self.estimators:
            raise ValueError("Estimator not fitted yet.")
        
        agg_pred = np.zeros(X.shape[0])
        for alpha, estimator in zip(self.alphas, self.estimators):
            pred = estimator.predict(X)
            pred = np.where(pred==0, -1, pred)
            agg_pred += alpha * pred

        y_pred = np.where(agg_pred>=0, self.labels[1], self.labels[0])

        return y_pred
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return deep copy of parameters.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            'estimator': self.estimator,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state
        }
        return copy.deepcopy(params) if deep else params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self

        Raises:
            ValueError: If an invalid parameter is passed.
        """
        for key, value in params.items():
            if key not in self.get_params(deep=False):
                raise ValueError(f"Invalid parameter '{key}' for estimator {self.__class__.__name__}")
            setattr(self, key, value)
        return self

