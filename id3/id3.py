import math

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

# length = len(data["play"])

class node:
    def __init__(self,col,val,arc):
        self.attr = attr          # attribute that was chosen to split
        self.val = val            # at the leafs of the tree, will be a bucket
        self.arc = arc

class arc:
    def __init__(self,attrVal,node):
        self.attrVal = attrVal
        self.node = node

class tree:
    def __init__(self,root):
        self.root = root

def calculateEntropy(data,outAttr):
    perType = {}
    for record in data:
        if perType.has_key(record[outAttr]):
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
        if perType.has_key(record[inAttr]):
            perType[record[inAttr]] += 1
        else:
            perType[record[inAttr]] = 1.0

    conditionalPairs = {}
    for record in data:
        key = (record[inAttr],record[outAttr])
        if conditionalPairs.has_key(key):
            conditionalPairs[key] += 1
        else:
            conditionalPairs[key] = 1.0

    ret = 0
    for x in conditionalPairs:
        ret -= perType[x[0]] / 14 * conditionalPairs[x] / perType[x[0]] * math.log(conditionalPairs[x] / perType[x[0]],2)

    return calculateEntropy(data,"play") - ret

# def divideOn(data,attr,value):

def id3():
    # print calculateEntropy(data,OUTPUT_ATTR)
    print calculateGain(data,"temperature",OUTPUT_ATTR)
    # divideOn(data,"temperature","cool")

id3()

