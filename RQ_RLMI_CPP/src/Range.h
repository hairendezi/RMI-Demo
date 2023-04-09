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
    inline bool match(const unsigned int &pos) const {
        if(pos >= this->Low && pos <= this->High) {
            return true;
        } else {
            return false;
        }
    }
};


#endif //RMIDEMO_RANGE_H
