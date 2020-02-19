import numpy as np
import random
import math
def initWeights(prevSize, nextSize, type = "random"):
    if type == "random":
        weights = np.zeros((nextSize,prevSize))
        for index, val in np.ndenumerate(weights):
            weights[index] = random.uniform(-5, 5)
        return weights

def calcWeightSum(layer, weights, nextLay, bias, act):
    for index,vec in enumerate(weights):
        weightSum = np.dot(layer,vec) + bias[index]
        actsum = nonLin(weightSum, act=act)
        nextLay[index] = actsum

def nonLin(num, act = "sig", deriv = False):
    if act == "sig":
        return sigmoid(num, deriv=deriv)
    elif act == "lekrelu":
        return leakyRelu(num, deriv=deriv)
    elif act == "relu":
        return relu(num, deriv=deriv)



def sigmoid(x, deriv=False):
    if (not deriv):
        return 1 / (1 + math.exp(-x))
    else:
        return sigmoid(x) * (1.0 - sigmoid(x))

def relu(x, deriv = False):
   if x > 0:
       if deriv:
        return 1
       return x
   else:
       return 0

def leakyRelu(x, deriv=False):
    if x > 0:
        if (not deriv):
            return x
        else:
            return 1
    else:
        if (not deriv):
            return x * 0.01
        else:
            return 0.01


class Layer:
    def __init__(self, size, act = "sig"):
        self.size = size
        self.vals = np.zeros(size)
        self.prevLayer = None
        self.nextLay = None
        self.weights = None
        self.bias = None
        self.act = act
        self.lastLay = None

    def addLayer(self, size, act = "sig"):
        newNext = Layer(size, act)
        if(self.nextLay is None):
            self.nextLay = newNext
            newNext.prevLayer = self
            self.weights = initWeights(self.size,newNext.size)
            self.bias = np.random.uniform(-5,5,size)
        else:
            end = self.nextLay
            while(end.nextLay is not None):
                end = end.nextLay
            end.nextLay = newNext
            newNext.prevLayer = end
            end.weights = initWeights(end.size,size)
            end.bias = np.random.uniform(-5,5, size)
            self.lastLay = newNext

    def feedForward(self, input):
        self.vals = input
        calcWeightSum(input,self.weights,self.nextLay.vals, self.bias, self.nextLay.act)
        lay = self.nextLay
        while(lay.nextLay is not None):
            calcWeightSum(lay.vals, lay.weights, lay.nextLay.vals, lay.bias, lay.nextLay.act)
            lay = lay.nextLay

    def backprop(self, target):
        for index, val in enumerate(self.lastLay.vals):
            self.lastLay.vals[index] = nonLin(val,self.lastLay.act,True)*(target[index] - val)
        currLayer = self.lastLay.prevLayer
        while(currLayer is not None):
            for index, val in enumerate(currLayer.vals):
                #print(type(currLayer))
                deltdotprod = np.dot(currLayer.nextLay.vals,currLayer.weights.transpose()[index])
                currLayer.vals[index] = nonLin(val,currLayer.act,True) * deltdotprod
            currLayer = currLayer.prevLayer
