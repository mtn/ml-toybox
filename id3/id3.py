import math
import json
from pprint import pprint
from collections import Counter

OUTPUT_ATTR = "play"


def calculateEntropy(data,outAttr):
    perType = {}
    for record in data:
        if record[outAttr] in perType:
            perType[record[outAttr]] += 1
        else:
            perType[record[outAttr]] = 1.0

    ret = 0
    for x in perType:
        freq = perType[x] / len(data)
        ret -= freq * math.log(freq,2)

    return ret

def calculateGain(data,inAttr,outAttr):
    perType = {}
    for record in data:
        if record[inAttr] in perType:
            perType[record[inAttr]] += 1
        else:
            perType[record[inAttr]] = 1.0

    conditionalPairs = {}
    for record in data:
        key = (record[inAttr],record[outAttr])
        if key in conditionalPairs:
            conditionalPairs[key] += 1
        else:
            conditionalPairs[key] = 1.0

    ret = 0
    for x in conditionalPairs:
        ret -= perType[x[0]] / 14 * conditionalPairs[x] / perType[x[0]] * math.log(conditionalPairs[x] / perType[x[0]],2)

    return calculateEntropy(data,"play") - ret

def determineSplit(data,inAttrs,outAttr):
    maxGain = 0
    splitAttr = inAttrs[0]
    for attr in set(inAttrs):
        gain = calculateGain(data,attr,outAttr)
        if gain > maxGain:
            maxGain = gain
            splitAttr = attr

    return splitAttr

def getMostFrequent(data,outArr):
    data = Counter([ var[outArr] for var in data ])
    return data.most_common(1)

def id3(data,outAttr,inAttrs):
    if len(data) == [ var[outAttr] for var in data ].count(data[0][outAttr]):
        return data[0][outAttr]
    if not data or len(inAttrs) <= 1:
        return getMostFrequent(data,outAttr)

    split = determineSplit(data,inAttrs,outAttr)
    tree = {split:{}}

    for val in set( [ op[split] for op in data ] ):
        subtree = id3( [ item for item in data if item[split] == val ],outAttr,[ attr for attr in inAttrs if attr != split ])
        tree[split][val] = subtree

    return tree

inAttrs = ["humidity","outlook","temperature","windy"]

def main():
    with open('data.json') as data_file:
        data = json.load(data_file)["data"]
    print(id3(data,OUTPUT_ATTR,inAttrs))

if __name__ == "__main__":
    main()

