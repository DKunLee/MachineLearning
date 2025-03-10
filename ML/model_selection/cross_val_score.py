import argparse
from typing import Tuple

from ML.model_selection.accuracy_score import accuracy_score

import pandas as pd
import numpy as np

def cross_val_score(model, X:pd.DataFrame, y:list, cv=5):
    shuffled_X, shuffled_y = shuffle_data(X, y)
    cv_indices = cv_folds(len(y), cv)

    scores = []

    for cv_index in cv_indices:
        train_X = shuffled_X.copy().drop(index=cv_index)
        train_y = [shuffled_y[i] for i in range(len(shuffled_y)) if i not in cv_index]
        test_X = shuffled_X.copy().iloc[cv_index]
        test_y = [shuffled_y[i] for i in cv_index]

        m = model

        m.fit(train_X, train_y)
        pred_y = m.predict(test_X)

        scores.append(accuracy_score(test_y, pred_y))
    
    return np.mean(scores)


def shuffle_data(X, y):
    df = X.copy()
    df['y_temp'] = y
    df_shuffled = df.sample(frac=1, random_state=None).reset_index(drop=True)
    y_shuffled = df_shuffled.pop('y_temp').tolist()
    return df_shuffled, y_shuffled


def cv_folds(len_of_data, cv):
    fold_sizes = [len_of_data // cv] * cv
    for i in range(len_of_data % cv):
        fold_sizes[i] += 1

    cv_indices = []
    st_idx = 0

    for size in fold_sizes:
        end_idx = st_idx + size
        cv_indices.append(list(range(st_idx, end_idx)))
        st_idx = end_idx
    
    return cv_indices