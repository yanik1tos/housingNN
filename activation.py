#!/usr/bin/env python3

import numpy as np


def f(z):
    return z

def f_prime(z, a):
    return np.ones_like(z)


def relu(z):
    return np.where(z > 0, z, 0)


def relu_prime(z, a):
    return np.where(z > 0, 1.0, 0)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z, a):
    # s = sigmoid(x)
    return (a * (1 - a)).reshape(-1, 1)


def softmax(z):
    z = z.copy()

    z = z - np.max(z)
    exp = np.exp(z)
    res = exp / np.sum(exp)

    return res


def cross_entropy(y, t):
    return -np.sum(t * np.log(y + 1e-12))
