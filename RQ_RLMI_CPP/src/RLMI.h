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

    inline int rqLookup(const unsigned long long int &key) const {
//        printf("lookup key: %llu\n", key);
        RLMINode *nowModel = stageModelList[0][0];
        RLMINode *tempModel;
        int baseIndex = 0;
        double output;
        for(int i=0; i<stageNum; i++) {
            output = nowModel->predict(key);
//            printf("a: %.3f, b: %.3f, output: %.3f, dataSize: %d, mu: %.3f, sig: %.3f\n",
//                   nowModel->_a, nowModel->_b, output, nowModel->dataSize, nowModel->mu, nowModel->sig);
//            for(int j=0; j<nowModel->dataSize; j++) {
//                nowModel->trainData[j]->printSelf();
////                printf("key: %.2f, value: %.2f\n", nowModel->keys[j], nowModel->values[j]);
//            }
//            printf("\n");
            if(output < 0) output = 0;
            if(output >= 1) output = 0.9999999;
            if(stageConfigList[i] != -1) {
//                output *= stageConfigList[i];
//                printf("next model index: %d\n", baseIndex + int(output*stageConfigList[i]));
                tempModel = this->stageModelList[i+1][baseIndex + int(output*stageConfigList[i])];
                if(tempModel->dataSize != 0) {
                    nowModel = tempModel;
                    baseIndex = (baseIndex + int(output*stageConfigList[i])) * stageConfigList[i];
                    continue;
                }
            }
            // Leaf search
            int searchBasePos = int(output * nowModel->dataSize);
            int start = searchBasePos - nowModel->maxOffset;
            int end = searchBasePos + nowModel->maxOffset + 1;
            if(start < 0) start = 0;
            if(end > nowModel->dataSize) end = nowModel->dataSize;
//            printf("start: %d, end: %d\n", start, end);
//            for(int j=start; j<end; j++) {
//                nowModel->trainData[j]->printSelf();
//            }
            if(nowModel->trainData[end-1]->rangeList[1]->match(key)) {
                return nowModel->trainData[end-1]->rangeList[1]->id;
            }
            for(int j=start; j<end; j++) {
                if(nowModel->trainData[j]->rangeList[0]->match(key)) return nowModel->trainData[j]->rangeList[0]->id;
            }
//            if(stageConfigList[i] != -1) {
//                output *= stageConfigList[i];
//                nowModel = this->stageModelList[i+1][baseIndex + int(output)];
//                baseIndex = (baseIndex + int(output)) * stageConfigList[i];
//            }
//            else {
//                int searchBasePos = int(output * nowModel->dataSize);
//                int start = searchBasePos - nowModel->maxOffset;
//                int end = searchBasePos + nowModel->maxOffset + 1;
//                if(start < 0) start = 0;
//                if(end > nowModel->dataSize) end = nowModel->dataSize;
//                if(nowModel->trainData[end-1]->rangeList[1]->match(key)) {
//                    return nowModel->trainData[end-1]->rangeList[1]->id;
//                }
//                for(int j=start; j<end; j++) {
//                    if(nowModel->trainData[j]->rangeList[0]->match(key)) return nowModel->trainData[j]->rangeList[0]->id;
//                }
//            }
        }
        return -1;
    }
};


#endif //RMIDEMO_RLMI_H
