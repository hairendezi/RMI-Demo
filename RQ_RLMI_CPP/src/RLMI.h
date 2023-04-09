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
//    std::vector<std::vector<RLMINode *> > stageModelList;
    std::vector<std::vector<std::vector<double> > > stageOutputList;

    RLMI(std::vector<KVEntry *> _trainData, int *_stageConfigList, int _stageNum);
    void build();
    int rqLookup(unsigned int key);
};


#endif //RMIDEMO_RLMI_H
