#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn_easy import NN_easy


reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
x0, y0 = reader.load_data_raw(features=1)
(x_train, y_train), (x_test, y_test) = reader.load_data(ratio=0.99, features=1)

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
plt.scatter(x, y)

plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))


nn = NN_easy(-1, -1, -1, 1)
nn.load("checkpoint/f1-214.492-2026-07-16 15:54:10.024265.npz")


x_nn = []
y_nn = []

for x in x_train:
    ans = nn.testOne(x, False)

    x_nn.append(int(reader.denormalize_x(x)[0][0]))
    y_nn.append(int(reader.denormalize_y(ans)[0][0]) / 1000)

print(x_nn[0])
print(y_nn[0])
print(reader.denormalize_y(y_test[0]) / 1000)

plt.plot(x_nn, y_nn, "red")


plt.xlabel("area m²")
plt.ylabel("price €k")
plt.grid("True")

# plt.savefig("results/graph_area_price.png", dpi=300)
plt.show()
