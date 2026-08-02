#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn import NN


def loss_for_weight(w_now):
    nn.weight[0][0][0] = w_now

    loss = 0

    for x_now, y_now in zip(x_train, y_train):
        ans = nn.testOne(x_now, False)[0][0]
        # print(ans, y_now)
        # exit(0)
        loss += 1/2 * (ans - y_now[0]) ** 2


    loss /= len(x_train)

    return loss


# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["area"]


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


x = x0
y = y0

plt.figure(figsize=(10, 8), dpi=150)

# plt.xlim(right=250)
# plt.ylim(bottom=0)


nn = NN.fromSave("checkpoint/f1_n0_ep300_b1_lr0.01-183.566-2026-07-28 11:43:06.343190.npz")

w = nn.weight[0][0][0]
print(w)

w_graph = np.linspace(0.5, 0.6, 100)
l_graph = []

for w_now in w_graph:
    l_graph.append(loss_for_weight(w_now))

l_graph = np.array(l_graph)

plt.plot(w_graph, l_graph, color="red", alpha=0.6, lw=2)

plt.xlim(w_graph.min(), w_graph.max())
plt.ylim(l_graph.min() - 0.0000005, l_graph.max())


# derivative
i = 1
for w in [0.51, 0.585, 0.5255, 0.565, 0.544]:
    m = (loss_for_weight(w + 0.0001) - loss_for_weight(w)) / 0.0001
    # print(m)
    # m = 2

    # d0_graph = np.linspace(w - 0.000002 / m, w + 0.000002 / m, 3)
    d0_graph = np.linspace(w - 0.004, w + 0.004, 3)
    l0_graph = []

    c = loss_for_weight(w) - w * m

    for w_now in d0_graph:
        # lw = loss_for_weight(w)
        # ln = loss_for_weight(w_now)
        l0_graph.append(m * w_now + c)

    plt.plot(d0_graph, l0_graph, color="black", alpha=0.75, lw=1.5)
    plt.scatter([w], [loss_for_weight(w)], alpha=1, color="black", s=10)
    plt.text(d0_graph[1] + 0.0011, l0_graph[1] + 0.00000002, str(i))

    i += 1


plt.xlabel("weight")
plt.ylabel("loss")
plt.grid("True")

# plt.savefig("results/how/graph.png", dpi=300)
plt.show()
