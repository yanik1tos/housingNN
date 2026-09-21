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


pathdata_2021 = "data/amsterdam_houses_all_distance.csv"


readData = DataLoaderCSV(pathdata_2021)
(x_train, y_train), (x_test, y_test) = readData.load_data(target=TARGET, features=FEATURES)



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
LOSS_FUNCTION = floss.mae
LOSS_FUNCTION_PRIME = floss.mae_prime


if LOSS_FUNCTION == floss.mse:
    loss_name = "MSE"
elif LOSS_FUNCTION == floss.mae:
    loss_name = "MAE"
elif LOSS_FUNCTION == floss.huber:
    loss_name = "HUBER"
else:
    loss_name = "IDK"


nn = NN(LEARNING_RATE, EPOCHS, BATCH_SIZE,
        INPUT_NODES, HIDDEN_LAYERS, HIDDEN_NODES, OUTPUT_NODES,
        ACTIVATION_HID, ACTIVATION_HID_PRIME,
        ACTIVATION_OUT, ACTIVATION_OUT_PRIME,
        loss_f=LOSS_FUNCTION, loss_f_prime=LOSS_FUNCTION_PRIME)



# train

nn.run(x_train, y_train)


# print(*[nn.weight[0][0][i] for i in range(len(FEATURES))])
# print(nn.bias[0][0][0])


nn.test(x_test, y_test, readData)
nn.save(pref=f"{loss_name}_n{HIDDEN_LAYERS}_ep{EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}")

