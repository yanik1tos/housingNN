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

        self.x_min = x.min(axis=0)
        self.x_max = x.max(axis=0)

        x = (x - self.x_min) / (self.x_max - self.x_min)

        return x


    def denormalize_x(self, x):
        x = x.reshape(
            -1, len(self.features)
        )

        return x * (self.x_max - self.x_min) + self.x_min


    def normalize_y(self, y):
        mn = y.min(axis=0)
        mx = y.max(axis=0)

        self.y_min = mn
        self.y_max = mx

        y = (y - mn) / (mx - mn)

        return y


    def denormalize_y(self, y):
        return y * (self.y_max - self.y_min) + self.y_min


    def load_data_raw(self,
                      target="price",
                      features=["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
                      ):
        data = pd.read_csv(self.path)

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        data = data.dropna(subset=features + [target])

        try:
            x = data[features].to_numpy(dtype=float)
            y = data[target].to_numpy(dtype=float)
        except:
            x = data[features].to_numpy()
            y = data[target].to_numpy()

        return x, y


    def load_data(self, ratio=0.95,
                  target="price",
                  features=["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
                  ):
        print("Loading the data in...")

        self.features = features

        x, y = self.load_data_raw(features=features)

        x = self.normalize_x(x)
        x = x.reshape(
            len(x), len(features), 1
        )

        y = self.normalize_y(y)
        y = y.reshape(
            len(y), 1
        )

        train_ln = int(len(x) * ratio)
        x_train = x[:train_ln]
        y_train = y[:train_ln]

        x_test = x[train_ln:]
        y_test = y[train_ln:]

        print("Data ready")

        print("Data_ln:", len(x))
        print("Train_ln:", train_ln)
        print("Test_ln:", len(x) - train_ln)

        print("X std:", round(x.std(), 3))
        print("X min max:", x.min(), x.max())

        print("Y std:", round(y.std(), 1))
        print("Y min max:", y.min(), y.max())

        print("X nan + inf:", np.isnan(x).sum(), np.isinf(x).sum())
        print("Y nan + inf:", np.isnan(y).sum(), np.isinf(y).sum())

        print()


        return (x_train, y_train), (x_test, y_test)



if "__main__" in __name__:
    pathdata_2021 = "data/amsterdam_houses_all_distance.csv"


    readData = DataLoaderCSV(pathdata_2021)
    (x_train, y_train), (x_test, y_test) = readData.load_data()

    np.set_printoptions(suppress=True, precision=2)

    print("X     :", *["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"])
    print("X mins:", readData.x_min)
    print("X maxs:", readData.x_max)
    print("Y mins: ", readData.y_min / 1000, "k €", sep="")
    print("Y maxs: ", readData.y_max / 1000, "k €", sep="")
