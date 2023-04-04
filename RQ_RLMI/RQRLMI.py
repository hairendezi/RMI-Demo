from DataLoader import *
from RQ_RLMI.RLMI import RLMI
from RQ_RLMI.Range import Range
from RQ_RLMI.KVClass import KVClass
import datetime

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
        startTime = datetime.datetime.now()
        self.rlmi.build()
        endTime = datetime.datetime.now()
        print("Build RLMI Cost Time: %.5f s" % (endTime-startTime).total_seconds())
        # self.rlmi.visualStageOutput()

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
        startPos, endPos, model = self.rlmi.lookup(pos)
        # print(startPos, endPos)
        for i in range(startPos, endPos):
            for j in model.trainData[i].rangeID:
                # print(j)
                if j.matchPos(pos):
                    return j.ID

if __name__ == '__main__':
    rangeData, posList = generateRangeQueryData(1000, [0, 65535])
    rqHandler = RangeQueryHandler(rangeData, posList)
    noneMatchDataCount = 0
    startTime = datetime.datetime.now()
    for i in range(0, 65536):
        if rqHandler.lookup(i) == None:
            noneMatchDataCount += 1
    endTime = datetime.datetime.now()
    print("Lookup keys Cost Time: %.5f s" % (endTime - startTime).total_seconds())
    print("None Match Data:", noneMatchDataCount)