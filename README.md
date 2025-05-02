# MachineLearning Scratch

A **lightweight**, **from-scratch** implementation of common machine learning algorithms and utilities in Python. Inspired by scikit-learn, **dklearn** provides simple, readable code for educational purposes and small-scale projects.

## 📦 Features

- **Linear Models**: Perceptron, Logistic Regression
- **Support Vector Machine**: Linear SVM (SVC)
- **Ensemble Methods**: AdaBoostClassifier
- **Decision Trees**: CART with Gini impurity
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Model Selection**: `train_test_split`, `cross_val_score`, `KFold`, `GridSearchCV`
- **Utilities**: `clone`, `shuffle_data`

*(Note: `dklearn` has no external dependencies beyond NumPy and joblib.)*

## 🎯 Usage

```python
import numpy as np
from dklearn.linear_model import Perceptron
from dklearn.ensemble import AdaBoostClassifier
from dklearn.metrics import accuracy_score, f1_score
from dklearn.model_selection import train_test_split

# 1. Prepare data
X = np.random.randn(200, 5)
y = np.random.randint(0, 2, size=200)

# 2. Split train/test
y_train, y_test = y[:150], y[150:]
X_train, X_test = X[:150], X[150:]

# 3. Train Perceptron
perceptron = Perceptron(max_iter=500, eta0=0.01, random_state=42)
perceptron.fit(X_train, y_train)

# 4. Evaluate
preds = perceptron.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("F1-score:", f1_score(y_test, preds))

# 5. Use AdaBoost with Perceptron base learner
adb = AdaBoostClassifier(estimator=perceptron, n_estimators=50, random_state=42)
adb.fit(X_train, y_train)
preds_adb = adb.predict(X_test)
print("AdaBoost F1:", f1_score(y_test, preds_adb))
```

## 📚 Structure Details

- **`dklearn.base`**
  - `clone(model)` - Deep-copy an estimator.
  - `shuffle_data(X, y)` - Shuffle features and labels consistently.

- **`dklearn.linear_model`**
  - `Perceptron(...)` - Binary perceptron classifier.
  - `LogisticRegression(...)` - SGD-based logistic regression with L2 regularization.

- **`dklearn.svm`**
  - `SVC(...)` - Linear SVM via SGD.

- **`dklearn.tree`**
  - `DecisionTreeClassifier(...)` - CART decision tree using Gini impurity.
  - `gini_impurity(y)`, `majority_label(y)` - Helpers.

- **`dklearn.ensemble`**
  - `AdaBoostClassifier(...)` - AdaBoost ensemble for binary classification.

- **`dklearn.metrics`**
  - `accuracy_score(y, pred_y)`, `precision_score(...)`, `recall_score(...)`, `f1_score(...)`.

- **`dklearn.model_selection`**
  - `train_test_split(...)`, `cross_val_score(...)`, `KFold`, `GridSearchCV`.

## 📊 Notebooks

This repository includes Jupyter notebooks demonstrating side-by-side comparisons between `dklearn` and `scikit-learn` implementations:

- `comparison_perceptron.ipynb`
- `comparison_logistic_regression.ipynb`
- `comparison_svc.ipynb`
- `comparison_adaboost.ipynb`
- `comparison_decision_tree.ipynb`

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📬 Contact

Created and maintained by DK Lee. For questions or suggestions, open an issue or contact via GitHub.

