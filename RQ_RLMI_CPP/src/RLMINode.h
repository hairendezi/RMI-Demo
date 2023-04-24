#ifndef RQ_RLMI_CPP_RLMINODE_H
#define RQ_RLMI_CPP_RLMINODE_H

#include <vector>
#include "KVEntry.h"

class RLMINode {
public:
    double _a;
    double _b;
    KVEntry **trainData;
    int dataSize;
    std::vector<unsigned int> _keys;
    std::vector<double> _values;
    std::vector<double> keys;
    std::vector<double> values;
    double mu;
    double sig;
    int maxOffset;

    RLMINode(KVEntry **trainData, int _dataSize);
    std::vector<double> build();
    void calMuSig();
    inline double predict(const unsigned long long int &key) const {
        double _key = (1.0 * key - this->mu) / this->sig;
        return this->_a * _key + this->_b;
    }
    void evaluateErrorBound();
};


#endif //RQ_RLMI_CPP_RLMINODE_H
