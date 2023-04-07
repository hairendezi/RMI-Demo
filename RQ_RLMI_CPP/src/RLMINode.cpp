#include "RLMINode.h"
#include <cmath>

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
        this->calMuSig();
        if(this->sig == 0) this->sig = 1;
        for(unsigned int key : this->_keys) {
            this->keys.push_back((1.0*key-this->mu) / this->sig);
        }

        // ===== Normalize Values in [0, 1] =====
        double minValue = this->_values[0], maxValue = this->_values[0];
        for(double value : this->_values) {
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
        double outputFactor = maxValue - minValue;
        if(outputFactor == 0) outputFactor = 1;
        for(double value : this->_values) {
            this->values.push_back((value - minValue) / outputFactor);
        }
    }
}

void RLMINode::calMuSig() {
    // calculate mean
    double sum = 0;
    for(unsigned int key : this->_keys) {
        sum += key;
    }
    this->mu = sum / this->dataSize;
    // calculate the sigma
    double sigma2 = 0;
    for(unsigned int key : this->_keys) {
        sigma2 += (1.0 * key - this->mu) * (1.0 * key - this->mu);
    }
    this->sig = sqrt(sigma2);
}

void RLMINode::build() {
    if(this->dataSize == 0) {
        this->_a = 0;
        this->_b = 0;
        return;
    }
    double keysAver = 0, valueAver = 0;
    double sigmaKV = 0, sigmaKK = 0;
    for(int i=0; i<this->dataSize; i++) {
        keysAver += this->keys[i];
        valueAver += this->values[i];
        sigmaKV += this->keys[i] * this->values[i];
        sigmaKK += this->keys[i] * this->keys[i];
    }
    keysAver /= this->dataSize;
    valueAver /= this->dataSize;

    if(this->dataSize * keysAver * keysAver == sigmaKK) {
        this->_a = 0;
        this->_b = 0;
    } else {
        this->_a = (sigmaKV - this->dataSize * keysAver * valueAver) / (sigmaKK - this->dataSize * keysAver * keysAver);
        this->_b = valueAver - this->_a * keysAver;
    }
}

void RLMINode::evaluateErrorBound() {
    if(this->dataSize == 0) return;
    std::vector<double> errorList;
    for(int i=0; i<this->dataSize; i++) {
        double value_hat = this->_a * this->keys[i] + this->_b;
        value_hat = std::max(std::min(value_hat, 0.9999999), 0.0);
        double predictPos = value_hat * this->dataSize;
        errorList.push_back(fabs(i-predictPos));
    }
    double maxError = errorList[0];
    for(double e : errorList) {
        maxError = std::max(e, maxError);
    }
    this->maxOffset = ceil(maxError);
}