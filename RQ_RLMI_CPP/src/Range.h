#ifndef RMIDEMO_RANGE_H
#define RMIDEMO_RANGE_H

#include <iostream>

class Range {
public:
    unsigned int Low;
    unsigned int High;
    unsigned int id;
    Range(unsigned int _Low, unsigned int _High, unsigned int _id);
    void printSelf();
    bool match(unsigned int pos);
};


#endif //RMIDEMO_RANGE_H
