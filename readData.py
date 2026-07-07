#!/usr/bin/env python3

import numpy as np
import pandas as pd


class DataLoaderCSV:
    def __init__(self, path):
        self.path = path


    def normalize_x(self, x):
        # mean = x.mean(axis=0)
        # std = x.std(axis=0)

        # x = (x - mean) / std

        mn = x.min(axis=0)
        mx = x.max(axis=0)

        x = (x - mn) / (mx - mn)

        return x

    def load_data(self, ratio=60):
        data = pd.read_csv(self.path)

        x = data[["Area", "Room", "Lon", "Lat"]].to_numpy(dtype=float)
        y = data["Price"].to_numpy(dtype=float)

        x = self.normalize_x(x)
        x = x.reshape(
            len(x), 4, 1
        )

        y = y / 1000
        y = y.reshape(
            len(y), 1
        )

        train_ln = len(data) // (ratio + 1) * ratio
        x_train = x[:train_ln]
        y_train = y[:train_ln]

        x_test = x[train_ln:]
        y_test = y[train_ln:]

        # print(len(data), train_ln, train_ln / (len(data) - train_ln))

        # print(x)
        # print(y)

        # print(x.std())
        # print(x.min(), x.max())


        return (x_train, y_train), (x_test, y_test)