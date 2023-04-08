#include <iostream>
#include <vector>
#include <string>
#include "./src/KVEntry.h"

std::vector<KVEntry> readData(char *filepath) {
    printf("===== Read Data =====\n");
    FILE *posFile = fopen(filepath, "r");
    int pos;
    std::vector<int> posList;
    while(true) {
        int temp = fscanf(posFile, "%d\n", &pos);
        printf("temp: %d\n", temp);
        if(temp != 1) {
            printf("%d\n", pos);
            break;
        }
        posList.push_back(pos);
    }
    for(int i : posList) {
        printf("%d\n", i);
    }
    return std::vector<KVEntry>{};
}

int main() {
    readData("../posdata.txt");
}