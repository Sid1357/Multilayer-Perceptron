# Multi-Layer Perceptron (MLP) for Titanic Survival Prediction

This repository contains an implementation of a multi-layer perceptron (MLP) from scratch for binary classification of Titanic survival using Python and NumPy. The model uses a custom-built neural network with backpropagation and gradient descent optimization. The Titanic dataset is used for training and testing the model's ability to predict survival outcomes.

## Project Overview

The project implements a simple feed-forward neural network for predicting Titanic survival. The dataset is preprocessed, and relevant features such as 'Pclass' and 'Sex' are used for classification. The model is trained using custom activation functions (ReLU and Sigmoid), with the backpropagation algorithm updating weights during training.

### Key Features:
- Data Preprocessing: Handling of missing values and categorical feature encoding.
- Neural Network Architecture: A multi-layer perceptron with two hidden layers.
- Custom Activation Functions: ReLU for hidden layers and Sigmoid for the output layer.
- Backpropagation: Manual implementation of the backpropagation algorithm to update weights.
- Performance Evaluation: Accuracy measurement on the test set.

## Mathematical Implementation

- **Activation Functions**: ReLU (Rectified Linear Unit) and Sigmoid were used to introduce non-linearity in the model. The derivative of these functions is computed for backpropagation.
- **Backpropagation**: The gradient of the loss function with respect to weights and biases is computed using the chain rule. This helps in updating the weights via gradient descent.
- **Gradient Descent Optimization**: Used to minimize the loss function and optimize the model's performance by adjusting the learning rate and weights iteratively.
- **Loss Function**: Mean squared error is used to measure the difference between predicted and actual outcomes during training.

## Dataset

The Titanic dataset (`train.csv`) consists of the following columns:

- `PassengerId`: Passenger identifier
- `Survived`: Survival status (0 = Died, 1 = Survived)
- `Pclass`: Passenger class (1 = First class, 2 = Second class, 3 = Third class)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger (male/female)
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Fare paid for the journey
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn (for splitting data and performance metrics)

## Results
-Accuracy: The model achieves an accuracy of approximately 72.07% on the test set
-Loss Visualization: The loss is plotted to observe the model’s convergence and optimization during training
