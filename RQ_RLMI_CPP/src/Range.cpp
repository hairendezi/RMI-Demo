#include "Range.h"

Range::Range(unsigned int _Low, unsigned int _High, unsigned int _id) {
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