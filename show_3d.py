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
train = train[:100]


nn = NN.fromSave("checkpoint/f1-183.834-2026-07-22 14:25:27.238174.npz", act=activation.f, act_prime=activation.f_prime)

print(nn.weight)
print(nn.bias)
# w1 = nn.weight[0][1][0]
# w2 = nn.weight[0][1][1]
# print(w1, w2)




# ranges for two weights
w1_values = np.linspace(-1, 1, 100)
w2_values = np.linspace(-1, 1, 100)

W1, W2 = np.meshgrid(w1_values, w2_values)

losses = np.zeros_like(W1)

for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        # temporarily change two weights
        nn.weight[0][0][0] = W1[i,j]
        nn.bias[0][0][0]   = W2[i,j]

        losses[i,j] = calc_loss()


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(W1, W2, losses)

ax.set_xlabel("weight")
ax.set_ylabel("bias")
ax.set_zlabel("loss")


# plt.savefig("results/how/graph_weight_bias_loss_sharp.png", dpi=300)
plt.show()
