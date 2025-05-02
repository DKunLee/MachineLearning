from dklearn.metrics import f1_score
from dklearn.metrics import accuracy_score
from dklearn.base import clone

import numpy as np
import itertools as it
from joblib import Parallel, delayed

def train_test_split(X: np.ndarray, y: np.ndarray, test_size:np.number=0.25, random_state:int=None, shuffle:bool=True):
    """
    Split arrays into random train and test subsets.

    Args:
        X (np.ndarray): Features, shape (n_samples, ...).
        y (np.ndarray): Labels, shape (n_samples,).
        test_size (float): Fraction for test split.
        random_state (int, optional): Seed.
        shuffle (bool): Shuffle before splitting.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            X_train, X_test, y_train, y_test
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(indices)

    test_count = int(n_samples * test_size)
    train_indices = indices[:-test_count]
    test_indices = indices[-test_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


def cross_val_score(estimator, X:np.ndarray, y:np.ndarray, scoring:str='accuracy_score', cv=None, random_state:int=None, shuffle:bool=False):
    """
    Evaluate a score by cross‑validation.

    Args:
        estimator: Any fitted‑style estimator (supports fit/predict).
        X (np.ndarray): Data to split.
        y (np.ndarray): Labels.
        scoring (str): 'accuracy_score' or 'f1_score'.
        cv (int or split‑generator): Number of folds or custom splitter.
        random_state (int): Seed for shuffle in CV.
        shuffle (bool): Shuffle data within CV.

    Returns:
        np.ndarray: Array of scores for each fold.
    """
    def get_result(estimator, X:np.ndarray, y:np.ndarray, train_idx, val_idx, scoring:str='accuracy_score'):
        X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]

        est_clone = clone(estimator)
        est_clone.fit(X_train, y_train)
        y_pred = est_clone.predict(X_val)

        if scoring == 'accuracy_score':
            score = accuracy_score(y_val, y_pred)
        elif scoring == 'f1_score':
            score = f1_score(y_val, y_pred)
        else:
            raise ValueError("Invalid scoring metric or hasn't implemented yet. Use {'accuracy_score', 'f1_score'}.")
        
        return score
    
    if cv is None:
        cv = KFold(n_splits=5, shuffle=shuffle, random_state=random_state)

    scores = Parallel(n_jobs=-1)(delayed(get_result)(estimator, X, y, train_idx, val_idx, scoring) for train_idx, val_idx in cv.split(X))

    return np.array(scores)


class KFold:
    """
    K fold cross validation splitter.
    """
    def __init__(self, n_splits:int=5, shuffle:bool=True, random_state:int=None):
        """
        Args:
            n_splits (int): Number of folds.
            shuffle (bool): Shuffle before splitting.
            random_state (int, optional): Seed.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X):
        """
        Generate indices for train/test splits.

        Args:
            X (array_like): Data to split.

        Yields:
            train_idx, val_idx: Numpy index arrays.
        """
        indices = np.arange(len(X))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, len(X)//self.n_splits, dtype=int)
        fold_sizes[:len(X)%self.n_splits] += 1
        curr = 0
        for fold_size in fold_sizes:
            val_idx = indices[curr:curr+fold_size]
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
            yield train_idx, val_idx
            curr += fold_size


class GridSearchCV:
    """
    Exhaustive search over specified parameter values for an estimator.
    """
    def __init__(self, estimator, param_grid:dict, scoring:str='accuracy_score', cv=5, n_jobs:int=-1, random_state:int=None, shuffle:bool=True):
        """
        Args:
            estimator: Base estimator.
            param_grid (dict): Parameter names mapped to lists of values.
            scoring (str): Metric for selecting best params.
            cv (int or splitter): Cross validation strategy.
            n_jobs (int): Parallel jobs.
            random_state (int, optional)
            shuffle (bool)
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.cv_results = {}
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.shuffle = shuffle
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_score_ = float('-inf')

    def fit(self, X:np.ndarray, y:np.ndarray):
        """
        Run fit with all parameter combinations and find the best.

        Args:
            X (np.ndarray)
            y (np.ndarray)
        """
        def get_result_of_cv(estimator, param_combi, scoring, cv):
            estimator_clone = clone(estimator)
            estimator_clone.set_params(**param_combi)
            scores = cross_val_score(estimator_clone, X, y, scoring=scoring, cv=cv, random_state=self.random_state, shuffle=self.shuffle)
            return scores, param_combi
        
        param_combinations = list(it.product(*self.param_grid.values()))
        param_combi_dict = [dict(zip(self.param_grid.keys(), values)) for values in param_combinations]

        if isinstance(self.cv, int):
            cv_object = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=self.shuffle)
        else:
            cv_object = self.cv

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(get_result_of_cv)(self.estimator, param_combi, self.scoring, cv_object) for param_combi in param_combi_dict
        )

        all_scores = []
        all_params = []

        for scores, param_combi in results:
            mean_score = np.mean(scores)
            all_scores.append(mean_score)
            all_params.append(param_combi)

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_parameters = param_combi

        self.cv_results['mean_test_scores'] = np.array(all_scores)
        self.cv_results['params'] = all_params
        self.cv_results['std_test_score'] = np.array([np.std(scores) for scores, _ in results])
        self.cv_results['rank_test_score'] = np.array([sorted(all_scores, reverse=True).index(x) + 1 for x in all_scores])

        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_parameters)

        self.best_estimator_.fit(X, y)

    def predict(self, X: np.ndarray):
        """
        Predict with the best found estimator.

        Args:
            X (np.ndarray)

        Returns:
            np.ndarray
        """
        if self.best_estimator_ is None:
            raise ValueError("You must call fit() before predict().")
        return self.best_estimator_.predict(X)

    def get_best_params(self):
        """Return best parameter combination found."""
        if self.best_parameters is None:
            raise ValueError("You must call fit() before get_best_params().")
        return self.best_parameters

    def get_best_estimator(self):
        """Return a refitted estimator with the best parameters."""
        if self.best_estimator_ is None:
            raise ValueError("You must call fit() before get_best_estimator().")
        return self.best_estimator_
    
    def get_cv_results(self):
        """Return summary of cross validation results."""
        if self.best_estimator_ is None:
            raise ValueError("You must call fit() before get_best_estimator().")
        return self.cv_results
