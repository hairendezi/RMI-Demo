#include "KVEntry.h"

KVEntry::KVEntry(unsigned long long int _key, double _value, Range **_rangeList) {
    this->key = _key;
    this->value = _value;
    this->rangeList = _rangeList;
}

void KVEntry::printSelf() {
    printf("key: %d, value: %.10f", key, value);
    rangeList[0]->printSelf();
    printf(", ");
    if(rangeList[1]) rangeList[1]->printSelf();
    printf("\n");
}