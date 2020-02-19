from layer import *

inp = np.array([1,2,3,4])
targ = np.array([0,1])
n = Layer(4)
n.addLayer(5, "relu")
n.addLayer(2)
n.feedForward(inp)
print(n.vals)
a = n.nextLay
while a is not None:
    print(a.vals)
    a = a.nextLay

