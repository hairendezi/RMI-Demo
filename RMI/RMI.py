import numpy
import torch

from RMINode import RMINode
from DataLoader import *
import matplotlib.pyplot as plt


class RMI:
    def __init__(self, networkStruct, stageConfigs: [], trainData):
        self.networkStruct = networkStruct
        self.stageConfigs = stageConfigs
        self.trainData = trainData


    def train(self):
        self.stageDataList = [[self.trainData]]
        self.modelList = []
        for i, stageConfig in enumerate(self.stageConfigs):
            self.modelList.append([])
            outputList = []
            for j, stageData in enumerate(self.stageDataList[i]):
                # 训练每个模型
                print("Start training on stage: %d model: %d" % (i, j))
                if len(stageData[0]) == 0:
                    print("no data, pass")
                    continue
                rmiNode = RMINode(self.networkStruct, stageConfig, stageData)

                trainCount = 0
                while True:
                    trainCount += 1
                    output, loss = rmiNode.train()
                    print("Train Count: %d, Loss: %.8f" % (trainCount, loss))
                    if loss < 0.001:
                        break

                outputList.append(output)
                self.modelList[i].append(rmiNode)

            # 如果不是最后一层，就计算下一层的数据分布，即计算stageDataList[i+1]
            if stageConfig["submodel_num"] != "leaf":
                self.stageDataList.append([])
                # 遍历这一层每一个子模型，划分这个模型对应的data成stageConfig["submodel_num"]份子数据供下一层训练
                for j, model in enumerate(self.modelList[i]):
                    if len(self.stageDataList[i][j][0]) == 0:
                        # print("no data")
                        continue
                    # print(np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, outputList[j])) * stageConfig["submodel_num"])
                    output = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, outputList[j])) * stageConfig["submodel_num"]
                    # print(list(outputList[j].reshape(1, -1)))
                    # print(outputList[j])
                    tempKeyList = [[] for _ in range(stageConfig["submodel_num"])]
                    tempValueList = [[] for _ in range(stageConfig["submodel_num"])]
                    for sdID, idx in enumerate(list(output)):
                        # print(int(idx), sdID, len(self.stageDataList[i][j][0]))
                        tempKeyList[int(idx)].append(self.stageDataList[i][j][0][sdID])
                        tempValueList[int(idx)].append(self.stageDataList[i][j][1][sdID])
                    self.stageDataList[i+1].extend([(numpy.array(tempKeyList[k]), numpy.array(tempValueList[k])) for k in range(stageConfig["submodel_num"])])
                    # print(tempKeyList)
                    # print(tempValueList)


    def test(self):
        keys, values = self.trainData
        # print(keys)
        ansList = [[] for _ in range(len(self.stageConfigs))]
        with torch.no_grad():
            for key in keys:
                # print("================================")
                baseIndex = 0
                nowModel = self.modelList[0][0]
                for i, stageConfig in enumerate(self.stageConfigs):
                    k = torch.tensor([(key - nowModel.mu) / nowModel.sig])
                    # print("look up key: %d, transform: " % (key), k, nowModel.mu, nowModel.sig)
                    if stageConfig["submodel_num"] == "leaf":
                        output = nowModel.net(k)
                        output = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, output))
                        # print(nowModel.loadData[1][int(output[0]*len(nowModel.loadData[0]))])
                        # print("stage: %d, last stage output: %.4f" % (i, output[0]))
                        ansList[i].append(nowModel.loadData[1][int(output[0]*len(nowModel.loadData[0]))])
                    else:
                        # 当前模型计算得到结构
                        output = nowModel.net(k)
                        # 选择下一层的模型
                        output = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, output))
                        ansList[i].append(nowModel.loadData[1][int(output[0] * len(nowModel.loadData[0]))])
                        output *= stageConfig["submodel_num"]
                        nowModel = self.modelList[i+1][baseIndex + int(output[0])]
                        # print("stage: %d, baseIndex: %d, next model: %d, output: %.4f" % (i, baseIndex, baseIndex + int(output[0]), output[0]))
                        baseIndex = (baseIndex + int(output[0])) * self.stageConfigs[i]["submodel_num"]
        # print(ansList)


        fig, ax = plt.subplots(nrows=len(self.stageConfigs), ncols=1, figsize=(5, 10))
        for i, stageData in enumerate(self.stageDataList):
            for data in stageData:
                ax[i].scatter(data[0], data[1], s=5)
                ax[i].plot(keys, ansList[i], c="orange")

        plt.show()

if __name__ == '__main__':
    networkStruct = [1, 8, 1]
    stageConfigs = [
        {
            "submodel_num": 4,
            "batch_size": 32,
            "lr": 0.01,
            "num_epochs": 100
        },
        {
            "submodel_num": 4,
            "batch_size": 32,
            "lr": 0.01,
            "num_epochs": 100
        },
        {
            "submodel_num": "leaf",
            "batch_size": 32,
            "lr": 0.01,
            "num_epochs": 150
        }
    ]
    trainData = generateRandomData(1500)
    rmi = RMI(networkStruct, stageConfigs, trainData)
    rmi.train()
    rmi.test()