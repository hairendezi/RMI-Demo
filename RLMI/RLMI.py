import DataLoader
from RLMINode import RLMINode
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import *


# Recursive Linear Model Index
class RLMI:
    def __init__(self, trainData, stageConfigList: []):
        self.trainData = trainData
        self.stageConfigList = stageConfigList

    def build(self):
        self.stageDataList = [[self.trainData]]
        self.stageModelList = []
        self.stageOutputList = []
        for i, stageConfig in enumerate(self.stageConfigList):
            stageData = self.stageDataList[i]
            stageModel = []
            stageOutput = []
            subStageData = []
            for j, modelData in enumerate(stageData):
                # if stageConfig["submodel_num"] != "leaf" and modelData is None or len(modelData[0]) == 0:
                #     subStageData.extend([None for _ in range(stageConfig["submodel_num"])])
                #     stageOutput.extend([None for _ in range(stageConfig["submodel_num"])])
                #     stageModel.append(None)
                #     continue
                # ===== Train Model =====
                rlmiNode = RLMINode(modelData)
                if not rlmiNode.noneData:
                    rlmiNode.build()
                stageModel.append(rlmiNode)
                output = rlmiNode.predictKeys(rlmiNode._keys)
                stageOutput.append(output)
                # rlmiNode.visualModel()

                # ===== Divide the Sub-Data =====
                if stageConfig["submodel_num"] != "leaf":
                    output = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, stageOutput[j])) * stageConfig["submodel_num"]
                    tempKeyList = [[] for _ in range(stageConfig["submodel_num"])]
                    tempValueList = [[] for _ in range(stageConfig["submodel_num"])]
                    for sdID, idx in enumerate(list(output)):
                        # print(int(idx), sdID, len(self.stageDataList[i][j][0]))
                        tempKeyList[int(idx)].append(stageData[j][0][sdID])
                        tempValueList[int(idx)].append(stageData[j][1][sdID])
                    subStageData.extend([(np.array(tempKeyList[k]), np.array(tempValueList[k])) for k in range(stageConfig["submodel_num"])])
                else:
                    rlmiNode.evaluateModel()
                    # print(tempKeyList)
                    # print(tempValueList)
            self.stageDataList.append(subStageData)
            self.stageModelList.append(stageModel)
            self.stageOutputList.append(stageOutput)

    def visualStageOutput(self):
        keys, values = self.trainData
        ansList = [[] for _ in range(len(self.stageConfigList))]
        noneMatchDataCount = 0
        for key in keys:
            lookTag = False
            nowModel = self.stageModelList[0][0]
            baseIndex = 0
            for i, stageConfig in enumerate(self.stageConfigList):
                output = nowModel.predict(key)
                # Normalize output in [0, 1]
                output = np.minimum(1 - np.finfo(np.float32).eps, np.maximum(0, output))
                ansList[i].append(nowModel._values[int(output * nowModel.dataSize)])
                # ===== Calculate Next Model Index =====
                if stageConfig["submodel_num"] != "leaf":
                    output *= stageConfig["submodel_num"]
                    nowModel = self.stageModelList[i + 1][baseIndex + int(output)]
                    baseIndex = (baseIndex + int(output)) * self.stageConfigList[i]["submodel_num"]
                # ===== Linear Search in Leaf Node =====
                else:
                    searchBasePos = int(output * nowModel.dataSize)
                    start = max(0, searchBasePos-nowModel.maxOffset-15)
                    end = min(nowModel.dataSize, searchBasePos+nowModel.maxOffset+15)
                    for idx in range(start, end):
                        # print(idx)
                        if nowModel._keys[idx] == key:
                            lookTag = True
                    if lookTag is False:
                        noneMatchDataCount += 1

        print("None Match Keys:", noneMatchDataCount)

        fig, ax = plt.subplots(nrows=len(self.stageConfigList), ncols=1, figsize=(5, 10))
        for i, stageData in enumerate(self.stageDataList):
            for data in stageData:
                ax[i].scatter(data[0], data[1], s=5)
                ax[i].plot(keys, ansList[i], c="orange")
        plt.savefig("./rlmi.pdf")
        plt.show()



if __name__ == '__main__':
    stageConfigList = [
        {"submodel_num": 4},
        {"submodel_num": 4},
        # {"submodel_num": 4},
        {"submodel_num": "leaf"}
    ]
    trainData = generateRandomData(1500)
    rlmi = RLMI(trainData, stageConfigList)
    rlmi.build()
    rlmi.visualStageOutput()
