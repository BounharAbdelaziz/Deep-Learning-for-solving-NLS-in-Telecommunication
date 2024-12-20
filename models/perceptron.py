from __future__ import print_function
from sklearn import datasets
import numpy as np

import sys
sys.path.append("C:/Users/admin/Desktop/IP Paris/MICAS/Cours/910/913 - Deep Learning/Project/")

# Import helper functions
from utils import train_test_split, normalize, to_categorical, accuracy_score
from networks.activation_functions import Sigmoid
from networks.loss_functions import CrossEntropy 
from utils import Plot
from networks import Perceptron


def main():
    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    # One-hot encoding of nominal y-values
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    # Perceptron
    clf = Perceptron(n_iterations=5000,
        learning_rate=0.001, 
        loss=CrossEntropy,
        activation_function=Sigmoid)
    clf.fit(X_train, y_train)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Perceptron", accuracy=accuracy, legend_labels=np.unique(y))


if __name__ == "__main__":
    main()