#!/usr/bin/env python3

import numpy as np
import random as r
import datetime


class NN_easy:
    def __init__(self, learning_rate, epochs, batch_size,
                 input_nodes):
        print("Initialization...\n")


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epoch = 0
        self.batch_size = batch_size

        self.input_nodes   = input_nodes
        self.output_nodes  = 1

        self.avg_error = None


        self.node  = [np.zeros((self.input_nodes , 1))] + \
                     [np.zeros((self.output_nodes, 1))]

        self.weight = [np.random.randn(self.output_nodes, self.input_nodes)  * np.sqrt(1 / self.input_nodes)]

        self.bias = [np.zeros((self.output_nodes, 1))]


        self.nmnx = [0, 0]


        print("Weight std:", round(self.weight[0].std(), 3))
        print("Weight min max:", round(self.weight[0].min(), 3), round(self.weight[0].max(), 3))


        print("Finished initialization\n")



    def calculate(self, test=False):
        # output layer
        self.node[1]  = self.weight[0] @ self.node[0] + self.bias[0]

        self.nmnx = [min(self.nmnx[1], float(self.node [1].min())),
                     max(self.nmnx[1], float(self.node [1].max()))]


        if test:
            print(
                f"Layer {1}: "
                f"z=[{self.node[1].min():.2f}, {self.node[1].max():.2f}] "
            )



    def learn(self, t, der_weight, der_bias):
        y = self.node[-1]

        loss = 1/2 * (y - t) ** 2

        dn = (y - t)

        der_weight[0] += dn * self.node[0].T
        der_bias[0]   += dn


        return loss


    def doEpoch(self, test=False):
        loss = 0
        batch = 0

        dw = [np.zeros_like(w) for w in self.weight]
        db = [np.zeros_like(b) for b in self.bias]


        for i in range(self.data_len):
            x = self.data_images[i]
            y = self.data_labels[i]

            self.node  = [np.zeros((self.input_nodes , 1))] + \
                         [np.zeros((self.output_nodes, 1))]


            if (i != self.data_len - 1):
                self.calculate()
            else:
                self.calculate(test=test)

            nloss = self.learn(y, dw, db)
            loss += nloss


            # print(dw[0] / self.batch_size)


            batch += 1

            if (batch == self.batch_size):
                batch = 0

                self.weight[0] -= dw[0] / self.batch_size * self.learning_rate
                self.bias[0]   -= db[0] / self.batch_size * self.learning_rate

                dw = [np.zeros_like(w) for w in self.weight]
                db = [np.zeros_like(b) for b in self.bias]


        if (self.data_len % self.batch_size != 0):
            self.weight[0] -= dw[0] / self.batch_size * self.learning_rate
            self.bias[0]   -= db[0] / self.batch_size * self.learning_rate



        # decay learning rate
        # self.learning_rate *= 0.99
        # self.learning_rate = max(0.001, self.learning_rate)


        loss /= self.data_len


        # print(self.weight)

        return loss


    def run(self, data_images, data_labels):
        self.data_len = len(data_labels)
        print(f"Unpacking data {self.data_len}...")

        self.data_labels = data_labels
        self.data_images = data_images

        print("Data prepared\n")


        print(f"Started learning for {self.epochs} epochs")


        while self.epoch < self.epochs:
            loss = self.doEpoch()
            self.epoch += 1


            if self.epoch != 0 and (self.epoch * 10) % self.epochs == 0:
                print(
                    f"Layer {1}: "
                    f"z=[{round(self.nmnx[0], 3)}, {round(self.nmnx[1], 3)}]"
                )

                self.nmnx = [0, 0]


                print(f"Epoch: {self.epoch * 100 // self.epochs}%")
                print(f"Learning rate: {self.learning_rate}")
                print("Loss: ", loss[0][0])


        print("Learning finished\n")


        print("Stats:\n")
        print("Out min:", round(self.node[-1].min(), 1))
        print("Out max:", round(self.node[-1].max(), 1))

        # print("\n###")

        # print("Weights: ", self.weight)
        # print("Biases: ", self.bias)

        # print("Nodes: ", self.node)
        # print("Anodes: ", self.anode)

        # print("###")

        print()



    def test(self, test_x, test_t, readData):
        test_len = len(test_x)

        print_i = [r.randint(0, test_len) for _ in range(10)]
        print_t = []
        print_y = []

        avg_loss = 0
        self.avg_error = 0
        avg_erper = 0

        for i in range(test_len):
            x = test_x[i]
            t = readData.denormalize_y(test_t[i][0]) / 1000

            y = readData.denormalize_y(self.testOne(x, test=(i == 0))[0][0]) / 1000

            avg_loss  += 1 / 2 * (y - t) ** 2
            self.avg_error += abs(y - t)
            avg_erper += abs(y - t) / t


            if (i in print_i):
                print_t.append(t)
                print_y.append(y)

            # print("---Test")
            # print("Target:", t)
            # print("Output:", round(y, 1))


        avg_loss  = avg_loss / test_len
        self.avg_error = self.avg_error / test_len
        avg_erper = avg_erper / test_len * 100

        print()
        print("avg_loss :", round(avg_loss, 1))
        print("avg_error:", round(self.avg_error, 1))
        print(f"avg_erper: {round(avg_erper, 1)}%")

        print()
        print("---Tests: ")
        for t, y in zip(print_t, print_y):
            print("Target:", t, "->", round(y, 1))



    def testOne(self, x, test):
        self.node  = [x] + \
                     [np.zeros((self.output_nodes, 1))]

        self.calculate(test=test)

        return self.node[-1]


    def save(self, pref="nw"):
        print(f"Saving the checkpoint...")
        name = f"checkpoint/{pref}-{str(round(self.avg_error, 3))}-{str(datetime.datetime.now())}"


        params = {
            "weight": np.array(self.weight, dtype=object),
            "bias": np.array(self.bias, dtype=object)
        }

        np.savez(name, **params)


        print("Saved at", name)


    def load(self, name):
        ckpt = np.load(name, allow_pickle=True)

        self.weight = ckpt["weight"]
        self.bias = ckpt["bias"]


