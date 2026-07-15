#!/usr/bin/env python3
#coding:utf

import numpy as np

from nn_easy import NN_easy
from readData import DataLoaderCSV



pathdata_2021 = "data/amsterdam_houses_all.csv"


readData = DataLoaderCSV(pathdata_2021)
(x_train, y_train), (x_test, y_test) = readData.load_data()



LEARNING_RATE = 0.0001
EPOCHS = 1000
BATCH_SIZE = 32

INPUT_NODES = 6


nn = NN_easy(LEARNING_RATE, EPOCHS, BATCH_SIZE, INPUT_NODES)



# train

# nn.run(x_train, y_train)
nn.load("checkpoint/ea-300.194-2026-07-09 22:37:09.742437.npz")

nn.test(x_test, y_test, readData)
nn.save(pref="ea")


print("Weight:", nn.weight)
print("Bias:", nn.bias)
print("X min max:", readData.x_min, readData.x_max)
print("Y min max:", readData.y_min, readData.y_max)

