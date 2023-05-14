#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <immintrin.h>
#include "./src/RLMI.h"

std::vector<KVEntry *> readData(char *filepath) {
    printf("===== Read Data =====\n");
    FILE *posFile = fopen(filepath, "r");
    unsigned long long int pos;
    std::vector<unsigned long long int> posList;
    while(true) {
        if(fscanf(posFile, "%llu\n", &pos) != 1) {break;}
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
    std::vector<KVEntry *> trainData = readData("D:\\Desktop\\RMIDemo\\random20.txt");
    printf("===== Start Build RLMI =====\n");
    auto startTime = std::chrono::high_resolution_clock::now();
//    int *stageConfigList = new int[4]{4, 4, 4, -1};
    int *stageConfigList = new int[4]{4, 4, -1};
    RLMI *rlmi = new RLMI(trainData, stageConfigList, 3);
    rlmi->build();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Build Time Cost: %lld ms\n", duration);
    printf("Model MAX Offset\n");
    int nodeNum = 1;
    for(int i=0; i<4; i++) {
        printf("Layer %d: ", i+1);
        for(int j=0; j<nodeNum; j++) {
            printf("%d ", rlmi->stageModelList[i][j]->maxOffset);
        }
        printf("\n");
//        if(i == 3) {
//            for(int j=0; j<96; j++) {
//                printf("model index: %d\n", j);
//                for(int k=0; k<rlmi->stageModelList[i][j]->dataSize; k++) {
//                    rlmi->stageModelList[i][j]->trainData[k]->printSelf();
//                }
//                printf("===========\n");
//            }
//        }
        nodeNum *= stageConfigList[i];
    }
    printf("Memory Consumption: %d byte\n", rlmi->evaluateMemory());

    printf("===== Start Match =====\n");
    int noneMatchCount = 0;
    startTime = std::chrono::high_resolution_clock::now();

//    rlmi->rqLookup(10000);
//    for(unsigned long long int i=2187654880;i<=2187654890;i++) {
//        if(i%1 == 0) {
//            printf("%llu\n", i);
//        }
//        if(rlmi->rqLookup(i) == -1) {
//            noneMatchCount ++;
//        }
//    }
    for(int N=0; N<100; N++) {
        for(unsigned long long int i=0; i<65536; i++) {
            if(rlmi->rqLookup(i) == -1) {
                noneMatchCount ++;
                printf("wrong index: %llu\n", i);
            }
        }
    }
    printf("\n");
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
    printf("Lookup Time Cost: %.6f ms\n", 0.01*duration/1e6);
    printf("None match count: %d\n", noneMatchCount);
//    printf("===== Start Test Pext Test =====\n");
//    startTime = std::chrono::high_resolution_clock::now();
//    long long int temp = 0;
//    for(long long int i=0; i<65536; i++) {
//        temp += _pext_u64(1234565, 12387866);
//    }
//    endTime = std::chrono::high_resolution_clock::now();
//    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
//    printf("Pext Instruction Time Cost: %lld ms, %lld\n", duration, temp);
}