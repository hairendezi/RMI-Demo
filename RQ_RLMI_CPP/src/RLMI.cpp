#include "RLMI.h"

RLMI::RLMI(std::vector<KVEntry> _trainData, std::vector<int> _stageConfigList) {
    this->trainData = _trainData;
    this->stageConfigList = _stageConfigList;
}


void RLMI::build() {
    stageDataList = {{this->trainData}};
    for(int i=0; i<this->stageConfigList.size(); i++) {
        int stageConfig = this->stageConfigList[i];
        std::vector<std::vector<KVEntry> > stageData = stageDataList[i];
        std::vector<RLMINode *> stageModel;
        std::vector<std::vector<double> > stageOutput;
        std::vector<std::vector<KVEntry> > subStageData;
        for(int j=0; j<stageData.size(); j++) {
            std::vector<KVEntry> modelData = stageData[j];
            RLMINode * rlmiNode = new RLMINode(modelData);
            stageOutput.push_back(rlmiNode->build());
            stageModel.push_back(rlmiNode);
            if(stageConfig != -1) {
                std::vector<std::vector<KVEntry> > dataElement(stageConfig);
                for(int sdID=0; i<stageOutput[j].size(); i++) {
                    double predictIndex = stageOutput[j][sdID];
                    dataElement[int(predictIndex)].push_back(stageData[j][sdID]);
                }
                for(std::vector<KVEntry> el : dataElement) {
                    subStageData.push_back(el);
                }
            }
            // Leaf Node, Need to evaluate the error bound
            else {
                rlmiNode->evaluateErrorBound();
            }
        }
        this->stageDataList.push_back(subStageData);
        this->stageModelList.push_back(stageModel);
        this->stageOutputList.push_back(stageOutput);
    }
}

int RLMI::rqLookup(unsigned int key) {
    RLMINode * nowModel = stageModelList[0][0];
    int baseIndex = 0;
    for(int i=0; i<this->stageConfigList.size(); i++) {
        int stageConfig = stageConfigList[i];
        double output = nowModel->predict(key);
        output = std::max(std::min(output, 0.9999999), 0.0);
        if(stageConfig != -1) {
            output *= stageConfig;
            nowModel = this->stageModelList[i+1][baseIndex + int(output)];
            baseIndex = (baseIndex + int(output)) * stageConfig;
        }
        else {
            int searchBasePos = int(output * nowModel->dataSize);
            int start = std::max(0, searchBasePos - nowModel->maxOffset);
            int end = std::min(nowModel->dataSize, searchBasePos + nowModel->maxOffset + 1);
            for(int j=start; j<end; j++) {
                for(Range & r : nowModel->trainData[j].rangeList) {
                    if(r.match(key)) {
                        return r.id;
                    }
                }
            }
        }
    }
    return -1;
}