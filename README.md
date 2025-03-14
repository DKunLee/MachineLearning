# Custom Machine Learning Modules

Welcome to the **Custom Machine Learning Modules** repository! This repository serves as a collection of various own implemented machine learning algorithms, and projects aimed at improving understanding and practical application of machine learning concepts.

## Table of Contents
- [Overview](#overview)
- [Structure](#structure)
- [Decision Tree Classifier](#decision-tree-classifier)
- [License](#license)

## Overview
This repository contains implementations of fundamental machine learning models using Python. It includes:
- Decision Tree Classifier
  
Upcoming:
- Perceptron
- ...

## Structure
```
ML/
│── model_selection/   
│──│──accuracy_score.py
│──│──cross_val_score.py
│──DecisionTreeClassifier.py
```

## Decision Tree Classifier
This model is the common decision tree classifier that classifies the numerical and categorical data.
Current implementation only has ID3 algorithm with 'information gain' with 'entropy'.

For categorical data:
- I just used the 'entropy' to find the 'information gain' and pick the best attrubutes among the subset.
  
For numerical data:
- I used 'potential split points' to find the best threshold and calculate the 'information gain'. If the best attribute is the numerical feature, then the node save the threshold.

## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Enjoy exploring my machine learning repo! 🚀

