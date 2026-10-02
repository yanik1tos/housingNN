#!/usr/bin/env python3

import numpy as np
import random as r
import datetime
import activation


r.seed(42)


class NN:
    def __init__(self, learning_rate, epochs, batch_size,
                 input_nodes, hidden_layers, hidden_nodes, output_nodes,
                 act, act_prime, outact, outact_prime):
        print("Initialization...\n")


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epoch = 0
        self.batch_size = batch_size

        self.input_nodes   = input_nodes
        self.hidden_layers = hidden_layers
        self.hidden_nodes  = hidden_nodes
        self.output_nodes  = output_nodes
        self.output_layer  = hidden_layers + 1

        self.act       = act
        self.act_prime = act_prime
        self.outact       = outact
        self.outact_prime = outact_prime

        self.avg_error = None


        self.node  = [np.zeros((self.input_nodes , 1))] + \
                     [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                     [np.zeros((self.output_nodes, 1))]

        self.anode = [np.zeros((self.input_nodes , 1))] + \
                     [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                     [np.zeros((self.output_nodes, 1))]


        if self.hidden_layers > 0:
            self.weight = [np.random.randn(self.hidden_nodes, self.input_nodes)  * np.sqrt(1 / self.input_nodes) ] + \
                        [np.random.randn(self.hidden_nodes, self.hidden_nodes) * np.sqrt(1 / self.hidden_nodes) for _ in range(self.hidden_layers - 1)] + \
                        [np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(1 / self.hidden_nodes)]


            self.bias = [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                        [np.zeros((self.output_nodes, 1))]
        else:
            self.weight = [np.random.randn(self.output_nodes, self.input_nodes) * np.sqrt(1 / self.input_nodes)]
            self.bias   = [np.zeros((self.output_nodes, 1))]


        self.nmnx = [[0, 0] for _ in range(self.hidden_layers + 2)]
        self.amnx = [[0, 0] for _ in range(self.hidden_layers + 2)]


        print("Weight std:", round(self.weight[0].std(), 3))
        print("Weight min max:", round(self.weight[0].min(), 3), round(self.weight[0].max(), 3))


        print("Finished initialization\n")


    @classmethod
    def fromSave(cls, path, learning_rate=0.001, epochs=10, batch_size=32,
                 act=activation.relu, act_prime=activation.relu_prime,
                 outact=activation.f, outact_prime=activation.f_prime):

        ckpt = np.load(path, allow_pickle=True)

        weight = ckpt["weight"]
        bias   = ckpt["bias"]


        input_nodes   = len(weight[0][0])
        hidden_nodes  = len(weight[0])
        hidden_layers = len(weight) - 1
        output_nodes  = len(weight[-1])

        # print(input_nodes, hidden_nodes, hidden_layers, output_nodes)


        model = cls(learning_rate, epochs, batch_size,
                    input_nodes, hidden_layers, hidden_nodes, output_nodes,
                    act, act_prime, outact, outact_prime)

        model.weight = weight
        model.bias   = bias

        return model



    def calculate(self, test=False):
        # hidden layer
        for i in range(1, self.output_layer):
            self.node[i]  = self.weight[i - 1] @ self.node[i - 1] + self.bias[i - 1]
            self.anode[i] = self.act(self.node[i])


            self.nmnx[i] = [min(self.nmnx[i][0], float(self.node [i].min())),
                            max(self.nmnx[i][1], float(self.node [i].max()))]
            self.amnx[i] = [min(self.amnx[i][0], float(self.anode[i].min())),
                            max(self.amnx[i][1], float(self.anode[i].max()))]

            if (test):
                print(
                    f"Layer {i}: "
                    f"z=[{self.node[i].min():.2f}, {self.node[i].max():.2f}] "
                    f"a=[{self.anode[i].min():.2f}, {self.anode[i].max():.2f}]"
                )

        # output layer
        i = self.output_layer

        self.node[i]  = self.weight[i - 1] @ self.node[i - 1] + self.bias[i - 1]
        self.anode[i] = self.outact(self.node[i])

        self.nmnx[i] = [min(self.nmnx[i][0], float(self.node [i].min())),
                        max(self.nmnx[i][1], float(self.node [i].max()))]
        self.amnx[i] = [min(self.amnx[i][0], float(self.anode[i].min())),
                        max(self.amnx[i][1], float(self.anode[i].max()))]


        if test:
            print(
                f"Layer {i}: "
                f"z=[{self.node[i].min():.2f}, {self.node[i].max():.2f}] "
                f"a=[{self.anode[i].min():.2f}, {self.anode[i].max():.2f}]"
            )



    def learn(self, t, der_weight, der_bias):
        y = self.anode[self.output_layer]

        loss = 1/2 * (y - t) ** 2

        dn = (y - t) * self.outact_prime(self.node[-1], self.anode[-1])

        # print()
        # print("y:", y)
        # print("t:", t)
        # print("dn:", dn)

        for i in range(self.hidden_layers, -1, -1):
            dw = dn * self.anode[i].T

            der_weight[i] += dw
            der_bias[i]   += dn

            dn = (self.weight[i].T @ dn) * self.act_prime(self.node[i], self.anode[i])


        return loss


    def doEpoch(self, test=False):
        loss = 0
        batch = 0

        dw = [np.zeros_like(w) for w in self.weight]
        db = [np.zeros_like(b) for b in self.bias]


        for i in range(self.data_len):
            x = self.data_images[i]
            y = self.data_labels[i]

            self.node = [x] + \
                        [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                        [np.zeros((self.output_nodes, 1))]

            self.anode = [x] + \
                         [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
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

                for i in range(len(self.weight)):
                    self.weight[i] -= dw[i] / self.batch_size * self.learning_rate

                for i in range(len(self.bias)):
                    self.bias[i]   -= db[i] / self.batch_size * self.learning_rate

                dw = [np.zeros_like(w) for w in self.weight]
                db = [np.zeros_like(b) for b in self.bias]


        if (self.data_len % self.batch_size != 0):
            for i in range(len(self.weight)):
                self.weight[i] -= dw[i] / (self.data_len % self.batch_size) * self.learning_rate

            for i in range(len(self.bias)):
                self.bias[i]   -= db[i] / (self.data_len % self.batch_size) * self.learning_rate



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
                for i in range(1, self.hidden_layers + 2):
                    print(
                        f"Layer {i}: "
                        f"z=[{self.nmnx[i][0]:.5f}, {self.nmnx[i][1]:.5f}] "
                        f"a=[{self.amnx[i][0]:.5f}, {self.amnx[i][1]:.5f}]"
                    )

                self.nmnx = [[0, 0] for _ in range(self.hidden_layers + 2)]
                self.amnx = [[0, 0] for _ in range(self.hidden_layers + 2)]


                print(f"Epoch: {self.epoch * 100 // self.epochs}%")
                print(f"Learning rate: {self.learning_rate}")
                print(f"Loss: {loss[0][0]:.10f}")


        print("Learning finished\n")


        print("Stats:\n")
        print("Out min:", round(self.anode[-1].min(), 1))
        print("Out max:", round(self.anode[-1].max(), 1))

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
            t = test_t[i][0]
            y = self.testOne(x, test=(i == 0))[0][0]

            t_n = readData.denormalize_y(t) / 1000
            y_n = readData.denormalize_y(y) / 1000

            avg_loss  += 1 / 2 * (y - t) ** 2
            self.avg_error += abs(y_n - t_n)
            avg_erper += abs(y_n - t_n) / t_n


            if (i in print_i):
                print_t.append(t_n)
                print_y.append(y_n)


        avg_loss  = avg_loss / test_len
        self.avg_error = self.avg_error / test_len
        avg_erper = avg_erper / test_len * 100

        print()
        print("avg_loss :", avg_loss)
        print("avg_error:", round(self.avg_error, 1))
        print(f"avg_erper: {round(avg_erper, 1)}%")

        print()
        print("---Tests: ")
        for t, y in zip(print_t, print_y):
            print("Target:", t, "->", round(y, 1))



    def testOne(self, x, test):
        self.node = [x] + \
                    [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                    [np.zeros((self.output_nodes, 1))]

        self.anode = [x] + \
                     [np.zeros((self.hidden_nodes, 1)) for _ in range(self.hidden_layers)] + \
                     [np.zeros((self.output_nodes, 1))]

        self.calculate(test=test)

        return self.anode[-1]


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
