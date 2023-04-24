#include "Range.h"

Range::Range(unsigned long long int _Low, unsigned long long int _High, unsigned long long int _id) {
    this->Low = _Low;
    this->High = _High;
    this->id = _id;
}

void Range::printSelf() {
    printf("Range %d: [%d, %d)", id, Low, High+1);
}

//bool Range::match(unsigned int pos) {
//    if(pos >= this->Low && pos <= this->High) {
//        return true;
//    } else {
//        return false;
//    }
//}