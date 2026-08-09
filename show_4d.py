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


nn = NN.fromSave("checkpoint/f3_n0_ep1000_b1_lr0.01-130.195-2026-08-07 21:53:43.941127.npz", act=activation.f, act_prime=activation.f_prime)

print(nn.weight)
print(nn.bias)
# w1 = nn.weight[0][1][0]
# w2 = nn.weight[0][1][1]
# print(w1, w2)


# Generate 3D points
x = np.linspace(0, 1, 20)
y = np.linspace(0, 1, 20)
z = np.linspace(0, 1, 20)

X, Y, Z = np.meshgrid(x, y, z)

w1 = nn.weight[0][0][0]
w2 = nn.weight[0][0][1]
w3 = nn.weight[0][0][2]
b = nn.bias[0][0]

W = X * w1 + Y * w2 + Z * w3 + b

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Flatten everything into individual points
scatter = ax.scatter(
    X.flatten(),
    Y.flatten(),
    Z.flatten(),
    c=W.flatten(),
    cmap="viridis",
    s=5
)

ax.set_xlabel("area")
ax.set_ylabel("year")
ax.set_zlabel("distance")

# x-y
# ax.view_init(elev=90, azim=-90)

# y-z
# ax.view_init(elev=0, azim=0)

# x-z
ax.view_init(elev=0, azim=90)

fig.colorbar(scatter)

plt.show()
