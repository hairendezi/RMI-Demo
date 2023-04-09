#ifndef RMIDEMO_KVENTRY_H
#define RMIDEMO_KVENTRY_H

#include <iostream>
#include <vector>
#include "Range.h"

class KVEntry {
public:
    unsigned int key;
    double value;
    Range **rangeList;
    KVEntry(unsigned int _key, double _value, Range **_rangeList);
    void printSelf();
};


#endif //RMIDEMO_KVENTRY_H
