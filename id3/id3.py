import math
from collections import Counter

OUTPUT_ATTR = "play"
data = [
         {"humidity": "high", "outlook": "sunny", "play": "no", "temperature": "hot", "windy": "false"},
         {"humidity": "high", "outlook": "sunny", "play": "no", "temperature": "hot", "windy": "true"},
         {"humidity": "high", "outlook": "overcast", "play": "yes", "temperature": "hot", "windy": "false"},
         {"humidity": "high", "outlook": "rainy", "play": "yes", "temperature": "mild", "windy": "false"},
         {"humidity": "normal", "outlook": "rainy", "play": "yes", "temperature": "cool", "windy": "false"},
         {"humidity": "normal", "outlook": "rainy", "play": "no", "temperature": "cool", "windy": "true"},
         {"humidity": "normal", "outlook": "overcast", "play": "yes", "temperature": "cool", "windy": "true"},
         {"humidity": "high", "outlook": "sunny", "play": "no", "temperature": "mild", "windy": "false"},
         {"humidity": "normal", "outlook": "sunny", "play": "yes", "temperature": "cool", "windy": "false"},
         {"humidity": "normal", "outlook": "rainy", "play": "yes", "temperature": "mild", "windy": "false"},
         {"humidity": "normal", "outlook": "sunny", "play": "yes", "temperature": "mild", "windy": "true"},
         {"humidity": "high", "outlook": "overcast", "play": "yes", "temperature": "mild", "windy": "true"},
         {"humidity": "normal", "outlook": "overcast", "play": "yes", "temperature": "hot", "windy": "false"},
         {"humidity": "high", "outlook": "rainy", "play": "no", "temperature": "mild", "windy": "true"}
]


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
    print(inAttrs)
    for attr in set(inAttrs):
        gain = calculateGain(data,attr,outAttr)
        print("gain " + str(gain) + " attr " + attr)
        if gain > maxGain:
            maxGain = gain
            splitAttr = attr

    return splitAttr

def getMostFrequent(data,outArr):
    data = Counter([ var[outArr] for var in data ])
    return data.most_common(1)  # Returns the highest occurring item

def id3(data,outAttr,inAttrs):
    if not data:
        return
    if len(data) == data.count(data[0][outAttr]):
        return data[0][outArr]
    if len(inAttrs) == 0:
        return getMostFrequent(data,outAttr)

    split = determineSplit(data,inAttrs,outAttr)
    tree = {split:{}}

    for val in set( [ op[split] for op in data ] ):
        subtree = id3( [ item for item in data if item[split] == val ],outAttr,[ attr for attr in inAttrs if attr != split ])
        tree[split][val] = subtree

    print(tree)
    print("*****\n")
    return tree

inAttrs = ["humidity","outlook","temperature","windy"]

def main():
    id3(data,OUTPUT_ATTR,inAttrs)

if __name__ == "__main__":
    main()

