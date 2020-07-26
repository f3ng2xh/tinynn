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
    x, y = load_digits(return_X_y=True)
    # x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(x.shape)
    # print(y[0])

    y_oh = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    print(y.shape)

    network = Network(64, [512, 256, 128], n_class=10)
    network.train(x[0:1000], y_oh[0:1000], 100, 0.0001)

    preds = network.predict(x[1001:1500])
    y_true = y[1001:1500]
    y_pred = np.argmax(preds, axis=1)

    print(accuracy_score(y_true, y_pred))
