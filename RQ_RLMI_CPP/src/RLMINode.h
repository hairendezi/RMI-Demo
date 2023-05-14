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
    std::vector<unsigned long long int> _keys;
    std::vector<double> _values;
    std::vector<double> keys;
    std::vector<double> values;
    double mu;
    double sig;
    int maxOffset = -1;

    RLMINode(KVEntry **trainData, int _dataSize);
    std::vector<double> build();
    void calMuSig();
    inline double predict(const unsigned long long int &key) const {
//        double _key = (1.0 * key - this->mu) * this->sig;
        return this->_a * (1.0 * key - this->mu) * this->sig + this->_b;
    }
    void evaluateErrorBound();
    inline void printSelf() {
        printf("dataSize: %d, mu: %.3f, sig: %.3f, a: %.3f, b: %.3f\n", dataSize, mu, sig, _a, _b);
    }
};


#endif //RQ_RLMI_CPP_RLMINODE_H
