#include "RLMINode.h"


RLMINode::RLMINode(std::vector<KVEntry> trainData) {
    this->_a = 0;
    this->_b = 0;
    this->trainData = trainData;
    this->dataSize = trainData.size();

    if(this->dataSize != 0) {
        for(auto d : this->trainData) {
            _keys.push_back(d.key);
            _values.push_back(d.value);
        }

        // ===== Normalize Keys in N(0, 1) =====



        // ===== Normalize Values in [0, 1] =====
    }
}