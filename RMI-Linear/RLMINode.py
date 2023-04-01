from DataLoader import *
import numpy as np
import matplotlib.pyplot as plt


# The node of Recursive Linear Model Index
# y = ax + b
class RLMINode:
    # Init the train data and the model param
    def __init__(self, trainData):
        self._a: float = 0
        self._b: float = 0
        self.trainData = trainData
        self._keys, self._values = self.trainData
        self.dataSize = len(self._keys)
        self.noneData = False

        if self.dataSize != 0:
            # ===== Normalize Keys in N(0, 1) =====
            self.mu = np.mean(self._keys)
            self.sig = np.std(self._keys)
            if self.sig == 0: self.sig = 1
            self.keys = (self._keys - self.mu) / self.sig

            # ===== Normalize Values in [0, 1] =====
            min_out = np.min(self._values)
            max_out = np.max(self._values)
            output_factor = (max_out - min_out)
            if output_factor == 0: output_factor = 1
            self.values = (self._values - min_out) / output_factor
        else:
            self.noneData = True

    # Calculate the linear Regression model
    def build(self):
        keysAver = np.mean(self.keys)
        valuesAver = np.mean(self.values)
        sigmaKV = 0
        sigmaKK = 0

        for i in range(self.dataSize):
            sigmaKV += self.keys[i] * self.values[i]
            sigmaKK += self.keys[i] ** 2

        self._a = (sigmaKV - self.dataSize * keysAver * valuesAver) / (sigmaKK - self.dataSize * keysAver * keysAver)
        self._b = valuesAver - self._a * keysAver

        # print(self._a, self._b)

    # output key's predict value
    def predict(self, _key: float):
        predictKey = (_key - self.mu) / self.sig
        return self._a * predictKey + self._b

    def predictKeys(self, _keys: []):
        retList = []
        for key in _keys:
            predictKey = (key - self.mu) / self.sig
            retList.append(self._a * predictKey + self._b)
        return retList

    # Visual the train data and the predict line model
    def visualModel(self):
        plt.figure(figsize=(4, 3))
        keys = self.keys
        values = self.values
        value_hats = []
        for i in range(self.dataSize):
            value_hats.append(self.predict(self._keys[i]))
        value_hats = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, value_hats))
        plt.scatter(keys, values, s=5)
        plt.plot(keys, value_hats, c="r")

        plt.show()

    def evaluateModel(self):
        for key, value in zip(self.keys, self.values):
            value_hat = self._a * key + self._b
            value_hat = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, value_hat))
            error = abs(value_hat-value)
            print(error)




if __name__ == '__main__':
    data = generateRandomData(1500)
    rlmi = RLMINode(data)
    rlmi.build()
    rlmi.visualModel()
    rlmi.evaluateModel()