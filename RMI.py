import numpy

from RMINode import RMINode
from DataLoader import *
import matplotlib.pyplot as plt


class RMI:
    def __init__(self, networkStruct, stageConfigs: [], trainData):
        self.networkStruct = networkStruct
        self.stageConfigs = stageConfigs
        self.trainData = trainData


    def train(self):
        fig, ax = plt.subplots(nrows=len(self.stageConfigs), ncols=1, figsize=(10, 20))

        self.stageDataList = [[self.trainData]]
        self.modelList = []
        for i, stageConfig in enumerate(self.stageConfigs):
            self.modelList.append([])
            outputList = []
            for j, stageData in enumerate(self.stageDataList[i]):
                # 训练每个模型
                print("Start training on stage: %d model: %d" % (i, j))
                ax[i].scatter(stageData[0], stageData[1], s=3)
                if len(stageData[0]) == 0:
                    print("no data, pass")
                    continue
                rmiNode = RMINode(self.networkStruct, stageConfig, stageData)
                output = rmiNode.train()

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
        plt.savefig("./result.pdf")
        plt.show()


    def test(self):
        pass


if __name__ == '__main__':
    networkStruct = [1, 8, 8, 1]
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