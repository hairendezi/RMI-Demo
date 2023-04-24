import numpy as np


def generateRandomData(dataSize: int):
    num_of_records = int(1e6)
    max_uint32 = np.iinfo(np.uint32).max

    database = np.random.lognormal(0, 1, num_of_records)
    # database = np.random.normal(0, 1, dataSize)
    # database = np.random.exponential(scale=0.3, size=100000)

    # 数据范围在[0-2^32-1]
    database -= np.min(database)
    database /= np.max(database)
    database *= max_uint32
    # 去重
    database = np.unique(database.astype(np.uint32))
    num_of_records = database.shape[0]

    # 生成 num_of_records * 2 形状的float类型数组，第0维初始化为上面生成的随机数据，第1维按比例归一化为0-1范围浮点数
    # dataset[:, 0] ==> keys | dataset[:, 1] ==> values
    dataset = np.zeros(shape=[num_of_records, 2]).astype(np.float32)
    dataset[:, 0] = database
    dataset[:, 1] = np.arange(num_of_records) / num_of_records
    dataset_indices = np.arange(num_of_records)
    sampledIndices = np.random.choice(dataset_indices, size=dataSize)
    sampledIndices.sort()
    print('** Generate Data: %d **' % dataSize)
    return dataset[sampledIndices, 0], dataset[sampledIndices, 1]


def generateRangeQueryData(posNum, dataRange):
    randomPosList = np.unique(np.sort(np.random.randint(dataRange[0], dataRange[1], posNum)))
    randomPosList = [pos for pos in randomPosList]
    randomPosList.insert(0, dataRange[0])
    randomPosList.append(dataRange[1]+1)
    rangeDataSet = []
    for i in range(len(randomPosList)-1):
        rangeDataSet.append([randomPosList[i], randomPosList[i+1]])

    return [
        {
            "LOW": rangeData[0],
            "HIGH": rangeData[1] - 1,
            "ID": id
        }
    for id, rangeData in enumerate(rangeDataSet)], randomPosList

def bugTest():
    posList = [0, 1056964608, 2684354560, 4009754624, 4294967296]
    rangeDataSet = []
    for i in range(len(posList) - 1):
        rangeDataSet.append([posList[i], posList[i + 1]])

    return [
        {
            "LOW": rangeData[0],
            "HIGH": rangeData[1] - 1,
            "ID": id
        }
        for id, rangeData in enumerate(rangeDataSet)], posList


if __name__ == '__main__':
    # print(generateRandomData(1500))
    # print(generateRangeQueryData(1000))
    # rangeData, posList = generateRangeQueryData(1000, [0, 65535])
    # with open("./posdata.txt", "w") as f:
    #     for i in posList:
    #         f.write(str(i)+"\n")
    print(bugTest())
