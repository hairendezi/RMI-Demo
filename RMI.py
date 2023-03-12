from RMINode import RMINode
from DataLoader import *


class RMI:
    def __init__(self, networkStruct, trainConfigs: [], trainData):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    networkStruct = [1, 8, 1]
    trainConfig = {
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 10
    }
    trainData = generateRandomData(1500)