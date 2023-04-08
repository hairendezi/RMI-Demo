#include <iostream>
#include <vector>
#include <string>
#include <chrono>
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
//    for(int i : posList) {
//        printf("%d\n", i);
//    }
    std::vector<Range> rangeList;
    for(int i=0; i<posList.size()-1; i++) {
        rangeList.push_back(Range(posList[i], posList[i+1], i));
    }
    double gapSize = 1.0 / posList.size();
    std::vector<KVEntry *> trainData;
    for(int i=0; i<posList.size(); i++) {
        std::vector<Range> rangeID;
        if(i == 0) {rangeID.push_back(rangeList[i]);}
        else if(i == posList.size()-1) {rangeID.push_back(rangeList[i-1]);}
        else {
            rangeID.push_back(rangeList[i]);
            rangeID.push_back(rangeList[i-1]);
        }
        KVEntry *kvEntry = new KVEntry(posList[i], i*gapSize, rangeID);
        trainData.push_back(kvEntry);
    }
    printf("Pos num: %lld\n", trainData.size());
    return trainData;
}

int main() {
    std::vector<KVEntry *> trainData = readData("D:\\Desktop\\RMIDemo\\posdata.txt");
    printf("===== Start Build RLMI =====\n");
    auto startTime = std::chrono::high_resolution_clock::now();
    RLMI *rlmi = new RLMI(trainData, std::vector<int>{4, 4, 4, -1});
    rlmi->build();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Build Time Cost: %lld ms\n", duration);
    printf("===== Start Match =====\n");
//    printf("Match result: %d", rlmi->rqLookup(10000));
    int noneMatchCount = 0;
    startTime = std::chrono::high_resolution_clock::now();
    for(int i=0; i<65536; i++) {
        if(rlmi->rqLookup(i) == -1) {
            noneMatchCount ++;
        }
    }
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    printf("Lookup Time Cost: %lld ms\n", duration);
    printf("None match count: %d\n", noneMatchCount);
}