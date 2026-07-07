#!/usr/bin/env python3
#coding:utf

import numpy as np

import activation
from nn import NN

from readData import DataLoaderCSV



pathdata_2021 = "data/HousingPrices-Amsterdam-August-2021.csv"


readData = DataLoaderCSV(pathdata_2021)
(x_train, y_train), (x_test, y_test) = readData.load_data()


print(x_train[0])
print(y_train[0])


LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 1024

INPUT_NODES   = 4
HIDDEN_LAYERS = 1
HIDDEN_NODES  = 5
OUTPUT_NODES  = 1

ACTIVATION_HID = activation.relu
ACTIVATION_OUT = activation.f
ACTIVATION_HID_PRIME = activation.relu_prime
ACTIVATION_OUT_PRIME = activation.f_prime


nn = NN(LEARNING_RATE, EPOCHS, BATCH_SIZE,
        INPUT_NODES, HIDDEN_LAYERS, HIDDEN_NODES, OUTPUT_NODES,
        ACTIVATION_HID, ACTIVATION_HID_PRIME,
        ACTIVATION_OUT, ACTIVATION_OUT_PRIME)



# train

nn.run(x_train, y_train)
nn.test(x_test, y_test)
# nn.save(pref="na")

