from model import BNNModel
from DataLoader import generateRandomData
from torch import nn, optim, tensor
import torch

def readData(path):
    with open(path, "r") as f:
        data = f.readlines()
    kvList = []
    for idx, key in enumerate(data):
        kvList.append({
            "key": int(key.strip()),
            "value": idx
        })
    return kvList

def dataDecomposition(kvList):
    for kv in kvList:
        print([int(bit) for bit in bin(kv["key"])[2:]])


# def train(net, trainData, batch_size, lr, num_epochs):
#     loss = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr)
#     # dataIter = data.DataLoader(trainData, batch_size, shuffle=True)
#
#     retLoss = None
#
#     print("** Start training **")
#     for epoch in range(num_epochs):
#         for i, (key, value) in enumerate(dataIter):
#             # 前向传播
#             value_hat = net(key)
#             # 计算损失
#             l = loss(value_hat, value)
#
#             # 反向传播和优化
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#
#             # 输出统计信息
#             if i == len(dataIter) - 1 and epoch == num_epochs - 1:
#                 # print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch + 1, num_epochs, l.item()))
#                 retLoss = l.item()
#                 # self.test()


if __name__ == '__main__':
    kvList = readData("../random20.txt")
    print(kvList)
    dataDecomposition(kvList)
    # bnnModel = BNNModel()
    # train(bnnModel, data, 32, 0.01, 20)

