#!/usr/bin/env python3
#coding:utf

import numpy as np

import activation
import loss as floss
from nn import NN

from readData import DataLoaderCSV


TARGET = "price"
# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["area", "year", "distance"]


ks = np.linspace(0.1, 0.4, 5)

EPOCHS       = 50
DIFF_RANDOMS = 50
DIFF_KS      = len(ks)

randoms = [np.random.randint(1, 1000) for i in range(DIFF_RANDOMS)]


pathdata_2021 = "data/amsterdam_houses_all_distance.csv"



def train(x_train, y_train, x_test, y_test, LOSS_FUNCTION, LOSS_FUNCTION_PRIME, HUBER_K):
    LEARNING_RATE = 0.01
    BATCH_SIZE = 1

    INPUT_NODES   = len(FEATURES)
    HIDDEN_LAYERS = 0
    HIDDEN_NODES  = 0
    OUTPUT_NODES  = 1

    ACTIVATION_HID = activation.f
    ACTIVATION_OUT = activation.f
    ACTIVATION_HID_PRIME = activation.f_prime
    ACTIVATION_OUT_PRIME = activation.f_prime


    nn = NN(LEARNING_RATE, EPOCHS, BATCH_SIZE,
            INPUT_NODES, HIDDEN_LAYERS, HIDDEN_NODES, OUTPUT_NODES,
            ACTIVATION_HID, ACTIVATION_HID_PRIME,
            ACTIVATION_OUT, ACTIVATION_OUT_PRIME,
            loss_f=LOSS_FUNCTION, loss_f_prime=LOSS_FUNCTION_PRIME,
            do_print=False, huber_k=HUBER_K)



    # train

    nn.run(x_train, y_train, do_print=False)

    avg_loss, avg_error, avg_erper = nn.test(x_test, y_test, readData, do_print=False)

    return avg_loss, avg_error, avg_erper


def do(HUBER_K):
    l = np.zeros((DIFF_RANDOMS, 3))

    for i in range(DIFF_RANDOMS):
        (x_train, y_train), (x_test, y_test) = readData.load_data(target=TARGET, features=FEATURES,
                                                                  rand=randoms[i], do_print=False)

        avg_loss, avg_error, avg_erper = train(x_train, y_train, x_test, y_test, floss.huber, floss.huber_prime, HUBER_K)

        l[i][0] = avg_loss
        l[i][1] = avg_error
        l[i][2] = avg_erper


    return np.mean(l, axis=0)



readData = DataLoaderCSV(pathdata_2021)

res = np.zeros((DIFF_KS, 3))

for i in range(DIFF_KS):
    HUBER_K = ks[i]
    res[i] = do(ks[i])

    print(f"Finshed {ks[i]}")


# print(res_mse)
# print(res_mae)
# print(res_huber)

print()
print()


np.set_printoptions(suppress=True)


for i in range(DIFF_KS):
    print(f"----------{ks[i]}")
    print(res[i])
