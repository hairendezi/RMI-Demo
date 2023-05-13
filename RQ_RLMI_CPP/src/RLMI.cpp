#include "RLMI.h"

RLMI::RLMI(std::vector<KVEntry *> _trainData, int *_stageConfigList, int _stageNum) {
    this->trainData = _trainData;
    this->stageConfigList = _stageConfigList;
    this->stageNum = _stageNum;
    this->stageModelList = new RLMINode**[stageNum];
}

void RLMI::build() {
    // init root data
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
            rlmiNode->evaluateErrorBound();
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
//            else {
//                rlmiNode->evaluateErrorBound();
//            }
        }
        this->stageDataList.push_back(subStageData);
        this->stageModelList[i] = stageModel;
        this->stageOutputList.push_back(stageOutput);
        stageModelNum *= stageConfig;
    }
}