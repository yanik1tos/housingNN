#!/usr/bin/env python3
#coding:utf

import numpy as np

import activation
from nn import NN

from readData import DataLoaderCSV


TARGET = "price"
# FEATURES = ["area", "energy", "rooms", "bedrooms", "bathrooms", "year", "distance"]
FEATURES = ["area", "year", "distance"]


pathdata_2021 = "data/amsterdam_houses_all_distance.csv"


readData = DataLoaderCSV(pathdata_2021)
(x_train, y_train), (x_test, y_test) = readData.load_data(target=TARGET, features=FEATURES)



LEARNING_RATE = 0.01
EPOCHS = 1000
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
        ACTIVATION_OUT, ACTIVATION_OUT_PRIME)



# train

# nn.run(x_train, y_train)
nn.load("checkpoint/f3_n0_ep1000_b1_lr0.01-130.195-2026-08-07 21:53:43.941127.npz")

# print(*[nn.weight[0][0][i] for i in range(len(FEATURES))])
# print(nn.bias[0][0][0])

nn.test(x_test, y_test, readData)
# nn.save(pref=f"f{len(FEATURES)}_n{HIDDEN_LAYERS}_ep{EPOCHS}_b{BATCH_SIZE}_lr{LEARNING_RATE}")

