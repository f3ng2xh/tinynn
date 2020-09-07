#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_digits
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from cnn_network import CnnNetwork
from cnn_network_lite import CnnNetwork
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    x, y = load_digits(return_X_y=True)
    #x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(x.shape)

    y_onehot = preprocessing.OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    print(y.shape)

    train_x = x[0:1700]
    train_y = y_onehot[0:1700]

    network = CnnNetwork(input_size=np.array([8, 8]), n_class=10)
    learning_rate = 0.005

    print("begin train ...")
    for epoch in range(4000):
        for i in range(2):
            xs = train_x[i].reshape((1, 8, 8))
            ys = train_y[i].reshape((10, 1))
            pred = network.predict_one_sample(xs)
            network.train_one_sample(ys, pred, learning_rate)
    print("end train ...")

    for i in range(2):
        preds = network.predict_one_sample(x[i].reshape((1, 8, 8)))
        print("preds:{}".format(preds))
        y_true = y[i]
        y_pred = np.argmax(preds, axis=0)
        print("label:{}, pred:{}".format(y_true, np.squeeze(y_pred)))

    # print(accuracy_score(y_true, y_pred))
