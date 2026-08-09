#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

from readData import DataLoaderCSV
from funda_collector.pdok import getLonLat

reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
adr, z = reader.load_data_raw(features=["address"])


adr = adr.reshape(-1)
z = z.reshape(-1) / 1000

x = np.zeros_like(adr)
y = np.zeros_like(adr)

for i in range(len(adr)):
    lon, lat = getLonLat(adr[i])

    x[i] = lon
    y[i] = lat




print("done")

# Create a regular grid
xi = np.linspace(x.min(), x.max(), 10)
yi = np.linspace(y.min(), y.max(), 10)
X, Y = np.meshgrid(xi, yi)

# Interpolate scattered points onto the grid
Z = griddata((x, y), z, (X, Y), method="linear")

# Contour plot
plt.contourf(X, Y, Z, levels=20)
plt.scatter(x, y, c=z, edgecolor="black", s=10)
plt.colorbar(label="z")
plt.show()
