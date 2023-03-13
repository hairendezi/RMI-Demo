import torch
from torch import nn, optim, tensor
import matplotlib.pyplot as plt
from DataLoader import *
import numpy as np


class RMINode:
    # networkStruct: [1, 8, 1]
    def __init__(self, networkStruct: [], trainConfig: {}, trainData):
        self.net = nn.Sequential()
        for i in range(len(networkStruct) - 1):
            self.net.append(nn.Linear(networkStruct[i], networkStruct[i + 1]))
            if i < len(networkStruct) - 2:
                self.net.append(nn.ReLU())

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_uniform_(m.bias, 0)

        self.net.apply(init_weights)
        # print("** Print Model **")
        # print(self.net)

        self.trainConfig = trainConfig
        self.preHandleData(trainData)

    def preHandleData(self, trainData):
        self.keys, self.values = trainData
        self.mu = np.mean(self.keys)
        self.sig = np.std(self.keys)
        # print((self.keys - mu) / sig)
        if self.sig == 0: self.sig = 1
        self.keys = (self.keys - self.mu) / self.sig
        self.keys = tensor(self.keys, dtype=torch.float32).reshape(-1, 1)

        # Calculate the output factor of the net
        min_out = np.min(self.values)
        max_out = np.max(self.values)
        output_factor = (max_out - min_out)
        if output_factor == 0: output_factor = 1
        # Normalize the net output to be in [0, 1]
        self.values = (self.values - min_out) / output_factor
        self.values = tensor(self.values, dtype=torch.float32).reshape(-1, 1)

        self.trainData = data.TensorDataset(self.keys, self.values)

    def train(self):
        batch_size = self.trainConfig["batch_size"]
        lr = self.trainConfig["lr"]
        num_epochs = self.trainConfig["num_epochs"]
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr)
        # optimizer = optim.SGD(self.net.parameters(), lr=lr)
        dataIter = data.DataLoader(self.trainData, batch_size, shuffle=True)

        # print("** Start training **")
        for epoch in range(num_epochs):
            # print(111)
            for i, (key, value) in enumerate(dataIter):
                # 前向传播
                value_hat = self.net(key)
                # 计算损失
                l = loss(value_hat, value)

                # 反向传播和优化
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                # 输出统计信息
                if i == len(dataIter) - 1 and epoch == num_epochs - 1:
                    print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, l.item()))
                    # self.test()

        with torch.no_grad():
            output = self.net(self.keys)
        return output.numpy()

    def test(self):
        plt.figure(figsize=(12, 4))
        plt.scatter(self.keys, self.values, s=1)
        with torch.no_grad():
            value_hats = self.net(self.keys)
            # print(value_hats)
            plt.plot(self.keys, value_hats.numpy(), c="r")
            plt.show()


if __name__ == '__main__':
    networkStruct = [1, 8, 1]
    trainConfig = {
        "batch_size": 32,
        "lr": 0.001,
        "num_epochs": 10
    }
    rmiNode = RMINode(networkStruct, trainConfig, generateRandomData(5000))
    rmiNode.train()
    rmiNode.test()