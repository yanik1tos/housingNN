#!/usr/bin/env python3

import matplotlib.pyplot as plt

from readData import DataLoaderCSV


reader = DataLoaderCSV("data/amsterdam_houses_all_distance.csv")
x0, y0 = reader.load_data_raw()

x0 = x0.reshape(
    len(x0), 7, 1
)
y0 = y0.reshape(
    len(y0), 1
)
y0 /= 1000


x = x0[:, 0]
y = y0

plt.figure(figsize=(10, 8), dpi=150)
plt.scatter(x, y)

# plt.xlim(0, 100)
# plt.ylim(0, 2000)

plt.xlabel("area m²")
plt.ylabel("price €k")
plt.grid("True")

# plt.savefig("results/graph_area_price.png", dpi=300)
plt.show()
