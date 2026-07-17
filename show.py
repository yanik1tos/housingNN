#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn import NN


# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["distance"]


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


x = x0
y = y0

plt.figure(figsize=(10, 8), dpi=150)

# plt.xlim(np.min(x), np.max(x))
# plt.ylim(np.min(y), np.max(y))


nn = NN.fromSave("checkpoint/f7-567.525-2026-07-16 16:32:58.262571.npz")


x_nn = []
y_nn = []

for x_now in x_train:
    ans = nn.testOne(x_now, False)

    x_nn.append(int(reader.denormalize_x(x_now)[0][0]))
    y_nn.append(int(reader.denormalize_y(ans)[0][0]) / 1000)


# print(reader.denormalize_y(y_test[0]) / 1000)


plt.scatter(x, y, color="blue", alpha=0.4)
plt.scatter(x_nn, y_nn, color="red", alpha=0.4)


plt.xlabel("area m²")
plt.ylabel("price €k")
plt.grid("True")

# plt.savefig("results/graph.png", dpi=300)
plt.show()
