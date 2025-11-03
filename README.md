# Machine Learning Algorithms from Scratch

A comprehensive, **from-scratch** implementation of fundamental machine learning algorithms in Python, demonstrating deep understanding of ML theory and NumPy-based computational methods. This library, **dklearn**, mirrors scikit-learn's API design while providing transparent, educational implementations of core algorithms.

## 🎯 Project Overview

This project showcases hands-on implementation of classical machine learning algorithms without relying on high-level ML frameworks. Each algorithm is built using only **NumPy** for numerical computations and **joblib** for parallel processing, emphasizing strong understanding of:

- Mathematical foundations of ML algorithms
- Object-oriented software design patterns
- Efficient vectorized operations with NumPy
- Parallel computing for model selection

## 🛠️ Tech Stack

**Core Technologies:**
- **Python 3.x** - Primary programming language
- **NumPy** - Numerical computing and vectorized operations
- **joblib** - Parallel processing for cross-validation and hyperparameter tuning
- **Jupyter Notebook** - Interactive demonstrations and comparisons

**Development Practices:**
- Object-oriented programming (OOP) with scikit-learn compatible API
- Clean code architecture with modular design
- Comprehensive documentation and type hints
- Comparative validation against scikit-learn benchmarks

## 📦 Implemented Algorithms

### Linear Models
- **Perceptron** - Binary linear classifier with SGD training ([linear_model.py:6-136](dklearn/linear_model.py#L6-L136))
- **Logistic Regression** - SGD-based probabilistic classifier with L2 regularization ([linear_model.py:138-285](dklearn/linear_model.py#L138-L285))

### Support Vector Machines
- **Linear SVM (SVC)** - Maximum margin classifier using subgradient descent ([svm.py:5-119](dklearn/svm.py#L5-L119))

### Tree-Based Models
- **Decision Tree Classifier** - CART algorithm with Gini impurity criterion ([tree.py:9-234](dklearn/tree.py#L9-L234))

### Ensemble Methods
- **AdaBoost Classifier** - Adaptive boosting with weighted voting ([ensemble.py:5-138](dklearn/ensemble.py#L5-L138))

### Model Selection & Validation
- **train_test_split** - Random data splitting utility
- **cross_val_score** - K-fold cross-validation with parallel execution
- **KFold** - K-fold splitter for cross-validation ([model_selection.py:83-120](dklearn/model_selection.py#L83-L120))
- **GridSearchCV** - Exhaustive hyperparameter search with parallelization ([model_selection.py:123-228](dklearn/model_selection.py#L123-L228))

### Evaluation Metrics
- **Accuracy Score** - Classification accuracy metric
- **Precision Score** - True positive ratio metric
- **Recall Score** - Sensitivity metric
- **F1 Score** - Harmonic mean of precision and recall ([metrics.py](dklearn/metrics.py))

## 🔬 Key Technical Implementations

**Optimization Algorithms:**
- Stochastic Gradient Descent (SGD) with learning rate decay
- Perceptron learning rule for linear separability
- Hinge loss optimization for SVM

**Ensemble Learning:**
- Adaptive sample weighting in AdaBoost
- Weighted majority voting for predictions

**Regularization Techniques:**
- L2 regularization in Logistic Regression
- Margin-based regularization in SVM

**Parallel Computing:**
- Multi-core processing for cross-validation folds
- Parallel grid search for hyperparameter optimization

## 💻 Usage Examples

### Basic Classification with Perceptron

```python
import numpy as np
from dklearn.linear_model import Perceptron
from dklearn.metrics import accuracy_score, f1_score
from dklearn.model_selection import train_test_split

# Prepare data
X = np.random.randn(200, 5)
y = np.random.randint(0, 2, size=200)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Perceptron
perceptron = Perceptron(max_iter=500, eta0=0.01, random_state=42)
perceptron.fit(X_train, y_train)

# Evaluate
predictions = perceptron.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"F1-Score: {f1_score(y_test, predictions):.4f}")
```

### Ensemble Learning with AdaBoost

```python
from dklearn.ensemble import AdaBoostClassifier
from dklearn.linear_model import Perceptron

# Create base estimator
base_estimator = Perceptron(max_iter=100, eta0=0.01, random_state=42)

# Train AdaBoost ensemble
ada_boost = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    random_state=42
)
ada_boost.fit(X_train, y_train)

# Predict and evaluate
predictions = ada_boost.predict(X_test)
print(f"AdaBoost F1-Score: {f1_score(y_test, predictions):.4f}")
```

### Hyperparameter Tuning with GridSearchCV

```python
from dklearn.model_selection import GridSearchCV
from dklearn.svm import SVC

# Define parameter grid
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 500, 1000]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=SVC(num_features=5),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy_score',
    n_jobs=-1
)

# Fit and find best parameters
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.get_best_params()}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
```

## 📊 Interactive Demonstrations

This repository includes Jupyter notebooks with **side-by-side comparisons** between `dklearn` and `scikit-learn`, validating implementation correctness and demonstrating equivalent performance:

- [Perceptron.ipynb](Perceptron.ipynb) - Perceptron implementation and validation
- [Logistic_Regression.ipynb](Logistic_Regression.ipynb) - Logistic regression with SGD
- [Support_Vector_Machine.ipynb](Support_Vector_Machine.ipynb) - Linear SVM comparison
- [Adaboost.ipynb](Adaboost.ipynb) - AdaBoost ensemble methods
- [Decsion_Tree.ipynb](Decsion_Tree.ipynb) - CART decision tree implementation

Each notebook demonstrates:
- Algorithm implementation details
- Performance comparison with scikit-learn
- Visualization of decision boundaries
- Hyperparameter sensitivity analysis

## 📂 Project Structure

```
MachineLearning/
├── dklearn/                      # Core library package
│   ├── base.py                   # Base utilities (clone, shuffle_data)
│   ├── linear_model.py           # Perceptron, Logistic Regression
│   ├── svm.py                    # Support Vector Machine
│   ├── tree.py                   # Decision Tree Classifier
│   ├── ensemble.py               # AdaBoost Classifier
│   ├── metrics.py                # Evaluation metrics
│   └── model_selection.py        # Cross-validation and grid search
├── Perceptron.ipynb              # Perceptron demonstrations
├── Logistic_Regression.ipynb     # Logistic regression examples
├── Support_Vector_Machine.ipynb  # SVM implementation details
├── Adaboost.ipynb                # Ensemble learning examples
├── Decsion_Tree.ipynb            # Decision tree visualizations
└── README.md                     # Project documentation
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **Machine Learning Theory**
   - Understanding of supervised learning algorithms
   - Knowledge of optimization techniques (SGD, gradient descent)
   - Ensemble learning and boosting methods

2. **Software Engineering**
   - Clean, maintainable code architecture
   - API design following established conventions (scikit-learn)
   - Comprehensive documentation and type annotations

3. **Computational Skills**
   - Efficient NumPy vectorization for performance
   - Parallel processing with joblib
   - Algorithm complexity analysis and optimization

4. **Research & Validation**
   - Empirical validation against established libraries
   - Comparative analysis and benchmarking
   - Interactive visualization and presentation

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.7+
python --version

# Install dependencies
pip install numpy joblib jupyter
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/MachineLearning.git
cd MachineLearning

# Import and use
python
>>> from dklearn.linear_model import Perceptron
>>> from dklearn.metrics import accuracy_score
```

### Running Notebooks
```bash
# Launch Jupyter Notebook
jupyter notebook

# Open any of the demonstration notebooks:
# - Perceptron.ipynb
# - Logistic_Regression.ipynb
# - Support_Vector_Machine.ipynb
# - Adaboost.ipynb
# - Decsion_Tree.ipynb
```

## 🔍 Why This Project Matters

**For Recruiters:**
- Demonstrates strong fundamentals in machine learning and computer science
- Shows ability to implement complex algorithms from mathematical specifications
- Exhibits clean code practices and software engineering skills
- Proves proficiency in Python, NumPy, and parallel computing

**For Graduate Admissions:**
- Evidence of deep understanding beyond using high-level libraries
- Research-oriented approach with empirical validation
- Self-directed learning and project completion
- Strong mathematical and computational foundations

## 📝 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📬 Contact

Created and maintained by **DK Lee**

For questions or suggestions, feel free to open an issue or reach out via GitHub.

---

**Note:** This is an educational project demonstrating algorithm implementations from scratch. For production use cases, please use established libraries like scikit-learn.