#include "RLMI.h"

RLMI::RLMI(std::vector<KVEntry *> _trainData, int *_stageConfigList, int _stageNum) {
    this->trainData = _trainData;
    this->stageConfigList = _stageConfigList;
    this->stageNum = _stageNum;
    this->stageModelList = new RLMINode**[stageNum];
}

void RLMI::build() {
    stageDataList = {{this->trainData}};
    int stageModelNum = 1;
    for(int i=0; i<this->stageNum; i++) {
        int stageConfig = this->stageConfigList[i];
        std::vector<std::vector<KVEntry *> > stageData = stageDataList[i];
        RLMINode **stageModel = new RLMINode*[stageModelNum];
        std::vector<std::vector<double> > stageOutput;
        std::vector<std::vector<KVEntry *> > subStageData;
        for(int j=0; j<stageData.size(); j++) {
            std::vector<KVEntry *> modelData = stageData[j];
            KVEntry **tempModelData = new KVEntry*[modelData.size()];
            for(int k=0; k<modelData.size(); k++) {tempModelData[k] = modelData[k];}

            RLMINode * rlmiNode = new RLMINode(tempModelData, modelData.size());
            stageOutput.push_back(rlmiNode->build());
            stageModel[j] = rlmiNode;
            if(stageConfig != -1) {
                std::vector<std::vector<KVEntry *> > dataElement(stageConfig);
                for(int sdID=0; sdID<stageOutput[j].size(); sdID++) {
                    double predictIndex = stageOutput[j][sdID] * stageConfig;
                    dataElement[int(predictIndex)].push_back(stageData[j][sdID]);
                }
                for(std::vector<KVEntry *> el : dataElement) {
                    subStageData.push_back(el);
                }
            }
            // Leaf Node, Need to evaluate the error bound
            else {
                rlmiNode->evaluateErrorBound();
            }
        }
        this->stageDataList.push_back(subStageData);
        this->stageModelList[i] = stageModel;
        this->stageOutputList.push_back(stageOutput);
        stageModelNum *= stageConfig;
    }
}

//int RLMI::rqLookup(const unsigned int &key) {
//    RLMINode * nowModel = stageModelList[0][0];
//    int baseIndex = 0;
//    for(int i=0; i<stageNum; i++) {
//        int stageConfig = stageConfigList[i];
//        double output = nowModel->predict(key);
//        if(output < 0) output = 0;
//        if(output >= 1) output = 0.9999999;
//        if(stageConfig != -1) {
//            output *= stageConfig;
//            nowModel = this->stageModelList[i+1][baseIndex + int(output)];
//            baseIndex = (baseIndex + int(output)) * stageConfig;
//        }
//        else {
//            int searchBasePos = int(output * nowModel->dataSize);
//            int start = searchBasePos - nowModel->maxOffset;
//            int end = searchBasePos + nowModel->maxOffset + 1;
//            if(start < 0) start = 0;
//            if(end > nowModel->dataSize) end = nowModel->dataSize;
//            if(nowModel->trainData[end-1]->rangeList[1]->match(key)) {
//                return nowModel->trainData[end-1]->rangeList[1]->id;
//            }
//            for(int j=start; j<end; j++) {
//                if(nowModel->trainData[j]->rangeList[0]->match(key)) return nowModel->trainData[j]->rangeList[0]->id;
//            }
//        }
//    }
//    return -1;
//}