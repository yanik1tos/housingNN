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


    def normalize_y(self, y):
        mn = y.min()
        mx = y.max()

        y = (y - mn) / (mx - mn)

        return y


    def load_data(self):
        print("Loading the data in...")

        data = pd.read_csv(self.path)

        features = ["Area", "Room", "Lon", "Lat"]
        target = "Price"

        data = data.dropna(subset=features + [target])

        x = data[features].to_numpy(dtype=float)
        y = data[target].to_numpy(dtype=float)

        x = self.normalize_x(x)
        x = x.reshape(
            len(x), 4, 1
        )

        y = y / 1000
        y = y.reshape(
            len(y), 1
        )

        train_ln = 800
        x_train = x[:train_ln]
        y_train = y[:train_ln]

        x_test = x[train_ln:]
        y_test = y[train_ln:]

        print("Data ready")

        print("Train_ln:", train_ln)
        print("Test_ln:", len(data) - train_ln)

        print("X std:", round(x.std(), 3))
        print("X min max:", x.min(), x.max())

        print("Y std:", round(y.std(), 1))
        print("Y min max:", y.min(), y.max())

        print("X nan + inf:", np.isnan(x).sum(), np.isinf(x).sum())
        print("Y nan + inf:", np.isnan(y).sum(), np.isinf(y).sum())

        print()


        return (x_train, y_train), (x_test, y_test)