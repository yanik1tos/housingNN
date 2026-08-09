#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn import NN


# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["area", "year", "distance"]


reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
x0, y0 = reader.load_data_raw(features=["distance"])
(x_train, y_train), (x_test, y_test) = reader.load_data(ratio=0.999, features=FEATURES)

x0 = x0.reshape(
    len(x0), 1, 1
)
y0 = y0.reshape(
    len(y0), 1
)
y0 /= 1000


plt.figure(figsize=(10, 8), dpi=150)

# plt.xlim(right=400)
plt.ylim(0, 7000)


nn = NN.fromSave("checkpoint/f3_n0_ep1000_b1_lr0.01-130.195-2026-08-07 21:53:43.941127.npz")

# print(nn.weight)
# print(nn.bias)

x_nn = []
y_nn = []

for x_now in x_train:
    ans = nn.testOne(x_now, False)

    x_nn.append(float(reader.denormalize_x(x_now)[0][2]))
    y_nn.append(float(reader.denormalize_y(ans)[0][0]) / 1000)


# print(reader.denormalize_y(y_test[0]) / 1000)

plt.scatter(x0, y0, color="blue", alpha=0.4, s=10)
plt.scatter(x_nn, y_nn, color="red", alpha=0.4, s=10)


plt.xlabel("distance km")
plt.ylabel("price €k")
plt.grid("True")

plt.savefig("results/nn/graph_distance_nn.png", dpi=300)
plt.show()
