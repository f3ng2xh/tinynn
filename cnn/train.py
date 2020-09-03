#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from cnn_network import CnnNetwork
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # x, y = load_digits(return_X_y=True)
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(x.shape)

    y_onehot = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    print(y.shape)

    network = CnnNetwork(input_size=np.array([28, 28]), n_class=10)
    learning_rate = 0.1
    print("begin train ...")
    for i in range(1000):
        xs = x[i].reshape((1, 28, 28))
        ys = y_onehot[i].reshape((10, 1))
        # print("ys:{}".format(ys))
        # print("y:{}".format(y[i]))
        pred = network.predict_one_sample(xs)
        network.train_one_sample(ys, pred, learning_rate)
    print("end train ...")

    for j in range(1):
        preds = network.predict_one_sample(x[j].reshape((1, 28, 28)))
        y_true = y[j]
        y_pred = np.argmax(preds, axis=0)
        print("label:{}, pred:{}".format(y_true, np.squeeze(y_pred)))

    # print(accuracy_score(y_true, y_pred))
