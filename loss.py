#!/usr/bin/env python3


import numpy as np


HUBER_K = 0.1



def mse(x):
    return 1/2 * x ** 2

def mse_prime(x):
    return x


def mae(x):
    return np.abs(x)

def mae_prime(x):
    return np.where(x < 0, -1, 1)


def huber(x):
    return np.where(np.abs(x) <= HUBER_K, mse(x),
                    HUBER_K * np.abs(x) - 1/2 * HUBER_K ** 2)

def huber_prime(x):
    return np.where(np.abs(x) <= HUBER_K, mse_prime(x),
                    HUBER_K * mae_prime(x))

