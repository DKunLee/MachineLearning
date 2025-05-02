import copy
import numpy as np

def clone(model):
    """
    Create a deep copy of the given model.

    Args:
        model: Any object implementing a `.copy()` or that can be deep_copied.

    Returns:
        A deep copy of `model`.
    """
    return copy.deepcopy(model)


def shuffle_data(X: np.ndarray, y: np.ndarray):
    """
    Randomly shuffle features and labels in unison along axis 0.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Label array of shape (n_samples,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: The shuffled (X, y).

    Raises:
        AssertionError: If `len(X) != len(y)`.
    """
    assert len(X) == len(y), f'{len(X)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(X))
    return X[p], y[p]
    