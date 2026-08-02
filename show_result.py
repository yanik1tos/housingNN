#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn import NN
import activation


FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
# FEATURES = ["area"]


reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
x0, y0 = reader.load_data_raw(features=["area"])
(x_train, y_train), (x_test, y_test) = reader.load_data(ratio=0.999, features=FEATURES)

x0 = x0.reshape(
    len(x0), 1, 1
)
y0 = y0.reshape(
    len(y0), 1
)
y0 /= 1000



plt.figure(figsize=(10, 8), dpi=150)


nn = NN.fromSave("checkpoint/f7_n0_ep1000_b32_lr0.01-128.474-2026-08-02 12:33:31.259567.npz", 
                 act=activation.f, act_prime=activation.f_prime)

for i in range(1, 6):
    nn.weight[0][0][i] = 0

predic = []
actual = []

for x, t in zip(x_train, y_train):
    y = nn.testOne(x, False)

    predic.append(int(reader.denormalize_y(y)[0][0]) / 1000)
    actual.append(int(reader.denormalize_y(t)[0]) / 1000)


# print(reader.denormalize_y(y_test[0]) / 1000)


plt.scatter(predic, actual, color="red", alpha=0.4, s=10)
plt.scatter(x0, y0 * nn.weight[0][0][0] + nn.bias[0][0], color="blue", alpha=0.4, s=10)

plt.xlabel("predicted")
plt.ylabel("actual")
plt.grid("True")

# plt.savefig("results/nn/graph_area_zoom.png", dpi=300)
plt.show()
