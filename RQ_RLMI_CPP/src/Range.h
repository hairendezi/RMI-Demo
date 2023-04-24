#ifndef RMIDEMO_RANGE_H
#define RMIDEMO_RANGE_H

#include <iostream>

class Range {
public:
    unsigned long long int Low;
    unsigned long long int High;
    unsigned long long int id;


    Range(unsigned long long int _Low, unsigned long long int _High, unsigned long long int _id);
    void printSelf();
    inline bool match(const unsigned long long int &pos) const {
        if(pos >= this->Low && pos <= this->High) {
            return true;
        } else {
            return false;
        }
    }
};


#endif //RMIDEMO_RANGE_H
