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


FEATURES = ["area", "year"]
reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
(x_train, y_train), (x_test, y_test) = reader.load_data(ratio=0.999, features=FEATURES)

train = x_train[:200]


plt.figure(figsize=(10, 8), dpi=150)


nn = NN.fromSave("checkpoint/f2_n0_ep1000_b1_lr0.01-149.489-2026-08-02 19:13:51.666382.npz", act=activation.f, act_prime=activation.f_prime)

print(nn.weight)
print(nn.bias)
# w1 = nn.weight[0][1][0]
# w2 = nn.weight[0][1][1]
# print(w1, w2)

a_graph = []
y_graph = []
p_graph = []


for x in train:
    x = reader.denormalize_x(x)
    a_graph.append(x[0][0])
    y_graph.append(x[1][0])


for a in train[:, 0, 0]:
    p_graph.append([])

    for y in train[:, 1, 0]:
        r = nn.testOne(np.array([a, y]), False)
        p_graph[-1].append(reader.denormalize_y(r)[0][0] / 1000)


# print(*a_graph[:10])
# print(*y_graph[:10])
# print(*p_graph[:10])


plt.contourf(a_graph, y_graph, p_graph, levels=100, cmap="plasma")


plt.xlabel("area")
plt.ylabel("year")
plt.colorbar(label="price")
plt.grid("True")

# plt.savefig("results/how/graph_weight_bias_loss_sharp.png", dpi=300)
plt.show()
