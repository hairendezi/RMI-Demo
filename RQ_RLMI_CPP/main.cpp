#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <immintrin.h>
#include "./src/RLMI.h"

std::vector<KVEntry *> readData(char *filepath) {
    printf("===== Read Data =====\n");
    FILE *posFile = fopen(filepath, "r");
    int pos;
    std::vector<int> posList;
    while(true) {
        if(fscanf(posFile, "%d\n", &pos) != 1) {break;}
        posList.push_back(pos);
    }
    std::vector<Range *> rangeList;
    for(int i=0; i<posList.size()-1; i++) {
        rangeList.push_back( new Range(posList[i], posList[i+1]-1, i));
    }
    double gapSize = 1.0 / posList.size();
    std::vector<KVEntry *> trainData;
    for(int i=0; i<posList.size(); i++) {
        Range **rangeID = new Range*[2];
        if(i == 0) {
            rangeID[0] = rangeList[i];
            rangeID[1] = rangeList[i];
        }
        else if(i == posList.size()-1) {
            rangeID[0] = rangeList[i-1];
            rangeID[1] = rangeList[i-1];
        }
        else {
            rangeID[0] = rangeList[i-1];
            rangeID[1] = rangeList[i];
        }
        KVEntry *kvEntry = new KVEntry(posList[i], i*gapSize, rangeID);
        trainData.push_back(kvEntry);
    }
    printf("Pos num: %lld\n", trainData.size());
    return trainData;
}

int main() {
    std::vector<KVEntry *> trainData = readData("D:\\Desktop\\RMIDemo\\bugdata.txt");
    printf("===== Start Build RLMI =====\n");
    auto startTime = std::chrono::high_resolution_clock::now();
    int stageConfigList[4] = {4, 4, 4, -1};
    RLMI *rlmi = new RLMI(trainData, stageConfigList, 4);
    rlmi->build();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Build Time Cost: %lld ms\n", duration);
    printf("===== Start Match =====\n");
    int noneMatchCount = 0;
    startTime = std::chrono::high_resolution_clock::now();
//    rlmi->rqLookup(2187654885);
//    rlmi->rqLookup(2187654886);
//    rlmi->rqLookup(2187654887);
    rlmi->rqLookup(2187654888);
//    rlmi->rqLookup(2187654889);
//    rlmi->rqLookup(2187654890);
//    for(unsigned long long int i=2187654880;i<=2187654890;i++) {
//        if(i%1 == 0) {
//            printf("%llu\n", i);
//        }
//        if(rlmi->rqLookup(i) == -1) {
//            noneMatchCount ++;
//        }
//    }
//    for(unsigned long long int i=0; i<4294967296; i++) {
//        if(i%10000000 == 0) {
//            printf("%llu\n", i);
//        }
//        if(rlmi->rqLookup(i) == -1) {
//            noneMatchCount ++;
//        }
//    }
//    printf("\n");
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
    printf("Lookup Time Cost: %.6f ms\n", 1.0*duration/1e6);
    printf("None match count: %d\n", noneMatchCount);
    printf("===== Start Test Pext Test =====\n");
//    startTime = std::chrono::high_resolution_clock::now();
//    long long int temp = 0;
//    for(long long int i=0; i<65536; i++) {
//        temp += _pext_u64(1234565, 12387866);
//    }
//    endTime = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
//    printf("Pext Instruction Time Cost: %lld ms, %lld\n", duration, temp);
}