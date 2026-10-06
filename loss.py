#!/usr/bin/env python3


import numpy as np


HUBER_K = 0.1



def mse(x, *args):
    return 1/2 * x ** 2

def mse_prime(x, *args):
    return x


def mae(x, *args):
    return np.abs(x)

def mae_prime(x, *args):
    return np.where(x < 0, -1, 1)


def huber(x, k=HUBER_K, *args):
    return np.where(np.abs(x) <= k, mse(x),
                    k * np.abs(x) - 1/2 * k ** 2)

def huber_prime(x, k=HUBER_K, *args):
    return np.where(np.abs(x) <= k, mse_prime(x),
                    k * mae_prime(x))
