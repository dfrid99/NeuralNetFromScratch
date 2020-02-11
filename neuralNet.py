import numpy as np
import math
import random


def sigmoid(x, deriv=False):
    if (not deriv):
        return 1 / (1 + math.exp(-x))
    else:
        return sigmoid(x) * (1.0 - sigmoid(x))

        # print(sigmoid(-55))

def leakyRelu(x, deriv = False):
    if x > 0:
        if(not deriv):
            return x
        else:
            return 1
    else:
        if (not deriv):
            return x * 0.01
        else:
            return 0.01


input_layer = np.array([2,1,1,1])
output_layer = np.array([0,1])

weights1 = np.zeros((2,4))
for index, val in np.ndenumerate(weights1):
    weights1[index] = random.uniform(-5,5)

#print(weights1.shape[0])

def feedforward(prev_layer, weights, bias):
    nextLay = np.zeros(weights1.shape[0])
    for index,vec in enumerate(weights):
        weightSum = np.dot(prev_layer,vec) + bias
        actSum = sigmoid(weightSum)
        nextLay[index] = actSum
    return nextLay

def calcDelta(layer, nextLay, outputLay = False):
    if(outputLay):
        for index,val in enumerate(layer):
            layer[index] = val * (1-val) * (val - nextLay[index])




bias = -5
a = feedforward(input_layer,weights1,bias)
print(a)
print(output_layer)
b = calcDelta(a, output_layer, outputLay=True)
print(a)
