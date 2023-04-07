#ifndef RQ_RLMI_CPP_RLMINODE_H
#define RQ_RLMI_CPP_RLMINODE_H

#include <vector>
#include "KVEntry.h"

class RLMINode {
public:
    double _a;
    double _b;
    std::vector<KVEntry> trainData;
    int dataSize;
    std::vector<unsigned int> _keys;
    std::vector<double> _values;
    std::vector<double> keys;
    std::vector<double> values;
    double mu;
    double sig;
    double maxOffset;

    RLMINode(std::vector<KVEntry> trainData);
    void build();
    void calMuSig();
    void evaluateErrorBound();
};


#endif //RQ_RLMI_CPP_RLMINODE_H
