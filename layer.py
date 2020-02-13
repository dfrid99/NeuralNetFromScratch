import numpy as np
import random
def initWeights(prevSize, nextSize, type = "random"):
    if(type == "random"):
        weights = np.zeros((nextSize,prevSize))
        for index, val in np.ndenumerate(weights):
            weights[index] = random.uniform(-5, 5)
        return weights

def calcWeightSum(layer, weights, nextLay):
    for index,vec in enumerate(weights):
        weightSum = np.dot(layer,vec)
        #actSum = sigmoid(weightSum)
        nextLay[index] = weightSum


class Layer:
    def __init__(self, size):
        self.size = size
        self.vals = np.zeros(size)
        self.prevLayer = None
        self.nextLay = None
        self.weights = None
        self.lastLay = None

    def addLayer(self, size):
        newNext = Layer(size)
        if(self.nextLay is None):
            self.nextLay = newNext
            newNext.prevLayer = self
            self.weights = initWeights(self.size,newNext.size)
        else:
            end = self.nextLay
            while(end.nextLay is not None):
                end = end.nextLay
            end.nextLay = newNext
            newNext.prevLayer = end
            end.weights = initWeights(end.size,size)
            self.lastLay = newNext

    def feedForward(self, input):
        self.vals = input
        calcWeightSum(input,self.weights,self.nextLay.vals)
        lay = self.nextLay
        while(lay.nextLay is not None):
            calcWeightSum(lay.vals, lay.weights, lay.nextLay.vals)
            lay = lay.nextLay









