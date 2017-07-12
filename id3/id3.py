import math

data = {
        "play": ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"],
        "windy": ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"],
        "outlook": ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"],
        "temperature": ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
        "humidity": ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
}

length = len(data["play"])

class node:
    def __init__(self,col,val,results,tNodes,fNodes):
        self.col = col
        self.val = val
        self.results = results
        self.tNodes = tNodes
        self.fNodes = fNodes

def calculateEntropy(data,outAttr):
    a = {}
    for val in data[outAttr]:
        if a.has_key(val):
            a[val] += 1
        else:
            a[val] = 1.0

    ret = 0
    for x in a:
        freq = a[x] / length
        ret -= freq * math.log(freq,2)

    return ret

def calculateGain(data,inAttr,outAttr):
    a = {}
    for val in data[inAttr]:
        if a.has_key(val):
            a[val] += 1
        else:
            a[val] = 1.0

    b = zip(data[outAttr],data[inAttr])
    c = {}
    for pair in b:
        if c.has_key(pair):
            c[pair] += 1
        else:
            c[pair] = 1.0

    ret = 0
    for x in c:
        ret -= a[x[1]] / 14 * c[x] / a[x[1]] * math.log(c[x] / a[x[1]],2)

    return calculateEntropy(data,"play") - ret

def id3():
    print calculateGain(data,"temperature","play")

id3()

