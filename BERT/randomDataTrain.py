from csv import reader as csv_reader
from os import mkdir

truePairs = []
falsePairs = []

with open("pairs/welfareTruePairs", "r+") as f:
    truePairsReader = csv_reader(f)

    for line in truePairsReader:
        truePairs.append(line)

with open("pairs/welfareRandomPairs", "r+") as f:
    falsePairsReader = csv_reader(f)

    for line in falsePairsReader:
        falsePairs.append(line)


for i, item in enumerate(truePairs):
    print(i)
    path = "welfareRandom/train/pos/{}.txt".format(i)
    with open(path, "w") as f:
        f.write(f"{item[0]} {item[1]}")


for i, item in enumerate(falsePairs):
    print(i)
    path = "welfareRandom/train/neg/{}.txt".format(i)
    with open(path, "w") as f:
        f.write(f"{item[0]} {item[1]}")
