#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from network import *
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    #x, y = load_digits(return_X_y=True)
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(x.shape)
    # print(y[0])

    y_oh = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(x, y_oh, test_size = 0.33, random_state = 42)

    #network = Network(64, [128, 64, 32], n_class=10)
    network = Network(784, [128, 64, 32], n_class=10)
    network.train(X_train, y_train, 10, 0.0001)

    y_preds = network.predict(X_test)
    print(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_preds, axis=1)))
