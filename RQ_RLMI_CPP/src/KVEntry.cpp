#include "KVEntry.h"

KVEntry::KVEntry(unsigned int _key, double _value, std::vector<Range> _rangeList) {
    this->key = _key;
    this->value = _value;
    this->rangeList = _rangeList;
}

void KVEntry::printSelf() {
    printf("key: %d, value: %.10f", key, value);
    for(Range &range : rangeList) {
        range.printSelf();
        printf(", ");
    }
    printf("\n");
}