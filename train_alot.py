#!/usr/bin/env python3
#coding:utf

import numpy as np

import activation
import loss as floss
from nn import NN

from readData import DataLoaderCSV



np.set_printoptions(suppress=True)


TARGET = "price"
# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["area", "year", "distance"]


pathdata_2021 = "data/amsterdam_houses_all_distance.csv"



def train(x_train, y_train, x_test, y_test, LOSS_FUNCTION, LOSS_FUNCTION_PRIME):
    LEARNING_RATE = 0.01
    EPOCHS = 100
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
            do_print=False)



    # train

    nn.run(x_train, y_train, do_print=False)

    avg_loss, avg_error, avg_erper = nn.test(x_test, y_test, readData, do_print=False)

    return avg_loss, avg_error, avg_erper


def do_loss(LOSS_FUNCTION, LOSS_FUNCTION_PRIME):
    n = 100
    l = np.zeros((n, 3))

    for i in range(n):
        (x_train, y_train), (x_test, y_test) = readData.load_data(target=TARGET, features=FEATURES,
                                                                  rand=np.random.randint(1, 100), do_print=False)

        avg_loss, avg_error, avg_erper = train(x_train, y_train, x_test, y_test, LOSS_FUNCTION, LOSS_FUNCTION_PRIME)

        l[i][0] = avg_loss
        l[i][1] = avg_error
        l[i][2] = avg_erper

    return l



readData = DataLoaderCSV(pathdata_2021)

res_mse   = do_loss(floss.mse  , floss.mse_prime)
print("Finshed MSE")
res_mae   = do_loss(floss.mae  , floss.mae_prime)
print("Finished MAE")
res_huber = do_loss(floss.huber, floss.huber_prime)
print("Finished HUBER")


# print(res_mse)
# print(res_mae)
# print(res_huber)

print()
print()

print(res_mse)
print(res_mae)
print(res_huber)

print()


print("----------MSE")
print("Mean:", np.mean(res_mse, axis=0))
print("Std :", np.std(res_mse, axis=0, ddof=1))
print("Min :", np.min(res_mse, axis=0))
print("Max :", np.max(res_mse, axis=0))

print("----------MAE")
print("Mean:", np.mean(res_mae, axis=0))
print("Std :", np.std(res_mae, axis=0, ddof=1))
print("Min :", np.min(res_mae, axis=0))
print("Max :", np.max(res_mae, axis=0))

print("----------HUBER")
print("Mean:", np.mean(res_huber, axis=0))
print("Std :", np.std(res_huber, axis=0, ddof=1))
print("Min :", np.min(res_huber, axis=0))
print("Max :", np.max(res_huber, axis=0))
