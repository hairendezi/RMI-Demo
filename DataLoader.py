import numpy as np
from torch.utils import data


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


def generateRangeQueryData(posNum):
    range = [0, 65535]
    randomPosList = np.unique(np.sort(np.random.randint(range[0], range[1], posNum)))
    rangeDataSet = []
    prePos = range[0]
    for pos in randomPosList:
        # [x, y)
        rangeDataSet.append([prePos, pos])
        prePos = pos
    rangeDataSet.append([prePos, range[1]])
    # print(rangeDataSet)
    return [
        {
            "LOW": rangeData[0],
            "HIGH": rangeData[1] - 1,
            "ID": id
        }
    for id, rangeData in enumerate(rangeDataSet)], [pos for pos in randomPosList]


if __name__ == '__main__':
    # print(generateRandomData(1500))
    print(generateRangeQueryData(1000))