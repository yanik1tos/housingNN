#!/usr/bin/env python3
#coding:utf

import numpy as np

import activation
from nn import NN

from readData import DataLoaderCSV



pathdata_2021 = "data/amsterdam_houses5000.csv"


readData = DataLoaderCSV(pathdata_2021)
(x_train, y_train), (x_test, y_test) = readData.load_data()



LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

INPUT_NODES   = 6
HIDDEN_LAYERS = 1
HIDDEN_NODES  = 10
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

# nn.run(x_train, y_train)
# nn.load("checkpoint/rl-371.495%-2026-07-08 23:25:34.556658.npz")

# nn.test(x_test, y_test, readData)
# nn.save(pref="rl")

