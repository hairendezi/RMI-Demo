#ifndef RMIDEMO_RLMI_H
#define RMIDEMO_RLMI_H

#include <vector>
#include "KVEntry.h"
#include "RLMINode.h"

class RLMI {
public:
    std::vector<KVEntry *> trainData;
    int *stageConfigList;
    int stageNum;
    RLMINode ***stageModelList;

    std::vector<std::vector<std::vector<KVEntry *> > > stageDataList;
    std::vector<std::vector<std::vector<double> > > stageOutputList;

    RLMI(std::vector<KVEntry *> _trainData, int *_stageConfigList, int _stageNum);
    void build();
    inline int rqLookup(const unsigned int &key) const {
        RLMINode * nowModel = stageModelList[0][0];
        int baseIndex = 0;
        for(int i=0; i<stageNum; i++) {
            int stageConfig = stageConfigList[i];
            double output = nowModel->predict(key);
            if(output < 0) output = 0;
            if(output >= 1) output = 0.9999999;
            if(stageConfig != -1) {
                output *= stageConfig;
                nowModel = this->stageModelList[i+1][baseIndex + int(output)];
                baseIndex = (baseIndex + int(output)) * stageConfig;
            }
            else {
                int searchBasePos = int(output * nowModel->dataSize);
                int start = searchBasePos - nowModel->maxOffset;
                int end = searchBasePos + nowModel->maxOffset + 1;
                if(start < 0) start = 0;
                if(end > nowModel->dataSize) end = nowModel->dataSize;
                if(nowModel->trainData[end-1]->rangeList[1]->match(key)) {
                    return nowModel->trainData[end-1]->rangeList[1]->id;
                }
                for(int j=start; j<end; j++) {
                    if(nowModel->trainData[j]->rangeList[0]->match(key)) return nowModel->trainData[j]->rangeList[0]->id;
                }
            }
        }
        return -1;
    }
};


#endif //RMIDEMO_RLMI_H
