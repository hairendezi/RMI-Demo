from DataLoader import *
from RQ_RLMI.RLMI import RLMI
from RQ_RLMI.Range import Range
from RQ_RLMI.KVClass import KVClass

class RangeQueryHandler:
    def __init__(self, rangeData, posList):
        self.preRangeHandler(rangeData, posList)
        self.stageConfigList = [
            {"submodel_num": 4},
            {"submodel_num": 4},
            {"submodel_num": 4},
            {"submodel_num": "leaf"}
        ]
        self.rlmi = RLMI(self.trainData, self.stageConfigList)
        self.rlmi.build()
        self.rlmi.visualStageOutput()

    def preRangeHandler(self, rangeData, posList):
        self.rangeDataList = [Range(rangeConfig) for rangeConfig in rangeData]
        self.dataSize = len(posList)
        self.rangeNum = len(self.rangeDataList)
        gapSize = 1 / self.dataSize
        self.trainData = []
        for i, pos in enumerate(posList):
            rangeID = []
            if i == 0:
                rangeID.append(self.rangeDataList[i])
            elif i == len(posList) - 1:
                rangeID.append(self.rangeDataList[i-1])
            else:
                rangeID.append(self.rangeDataList[i])
                rangeID.append(self.rangeDataList[i-1])
            tempKV = KVClass(pos, i * gapSize, rangeID)
            self.trainData.append(tempKV)

    def lookup(self, pos):
        startPos, endPos = self.rlmi.lookup(pos)
        for i in range(startPos, endPos):
            for j in self.trainData[i].rangeID:
                if j.match(pos):
                    return j.ID

if __name__ == '__main__':
    rangeData, posList = generateRangeQueryData(1000, [0, 65535])
    rqHandler = RangeQueryHandler(rangeData, posList)
    for i in range(0, 65536):
        # TODO: test the lookup result
        pass

