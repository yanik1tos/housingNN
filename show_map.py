#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from readData import DataLoaderCSV
from nn import NN
import activation


def calc_loss():
    loss = 0

    for x_now, y_now in train:
        ans = nn.testOne(x_now, False)[0][0]
        # print(ans, y_now)
        # exit(0)
        loss += 1/2 * (ans - y_now[0]) ** 2


    loss /= len(x_train)

    return loss


FEATURES = ["area"]
reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
(x_train, y_train), (x_test, y_test) = reader.load_data(ratio=0.999, features=FEATURES)

train = list(zip(x_train, y_train))
np.random.shuffle(train)
# train = train[:1000]


plt.figure(figsize=(10, 8), dpi=150)


nn = NN.fromSave("checkpoint/f1-183.834-2026-07-22 14:25:27.238174.npz", act=activation.f, act_prime=activation.f_prime)

print(nn.weight)
print(nn.bias)
# w1 = nn.weight[0][1][0]
# w2 = nn.weight[0][1][1]
# print(w1, w2)

w_graph = np.linspace(-1, 1, 30)
b_graph = np.linspace(-1, 1, 30)
l_graph = []

for w_now in w_graph:
    nn.weight[0][0][0] = w_now
    l_graph.append([])

    for b_now in b_graph:
        nn.bias[0][0][0] = b_now

        l_graph[-1].append(calc_loss())

l_graph = np.array(l_graph)

plt.contourf(w_graph, b_graph, l_graph, levels=100, vmin=0.0, vmax=0.1, cmap="plasma")
# plt.plot(w2_graph, l_graph, color="red")


plt.xlabel("weight")
plt.ylabel("bias")
plt.colorbar(label="loss")
plt.grid("True")

plt.savefig("results/how/graph_weight_bias_loss_sharp.png", dpi=300)
plt.show()
