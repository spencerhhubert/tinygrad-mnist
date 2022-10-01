from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from extra.utils import get_parameters
import tinygrad.nn.optim as optim
import numpy as np
import sys
import random
import math
sys.setrecursionlimit(10000)

from mnist import MNIST
mndata = MNIST("datasets/mnist")
mndata.gz = True

class NN():
    def __init__(self, input_size:int, sizes:list):
        dims = list(zip([input_size]+sizes[:-1], sizes))
        self.layers = [Linear(x,y) for x,y in dims]
    def forward(self,x):
        out = x
        for layer in self.layers:
            out = layer(out).relu()
        return out.logsoftmax()

class NN2():
    def __init__(self):
        self.l1 = Tensor.uniform(784, 128)
        self.l2 = Tensor.uniform(128, 10)

    def parameters(self):
        return get_parameters(self)

    def forward(self,x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

def labelVec(l:int):
    return [1 if x == l else 0 for x in range(10)] #0 except 1 at index of label

def shuffleData(data_tuple):
    images,labels = data_tuple
    temp = list(zip(images,labels))
    random.shuffle(temp)
    images, labels = zip(*temp)
    images, labels = list(images), list(labels)
    return (images, labels)

def getRandomBlock(data, size):
    images,labels = data
    random_idx = int(random.random()*len(images))
    st = random_idx
    sp = random_idx + size
    return (images[st:sp], labels[st:sp])

def sparse_categorical_crossentropy(out, Y):
    num_classes = out.shape[-1]
    YY = Y.flatten()
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),YY] = -1.0*num_classes
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return out.mul(y).mean()
    

nn = NN(784,[32,16,10])
tracked = []
for l in nn.layers:
    tracked.append(l.weight)
    tracked.append(l.bias)

model = NN2()

optimizer = optim.SGD(model.parameters(), lr=0.001)
batch_size = 64
epochs = 100

for i in range(epochs):
    images,labels = getRandomBlock(shuffleData(mndata.load_training()), batch_size)
    images = Tensor(images)
    #labels = Tensor(list(map(labelVec, labels)))
    labels = np.array(labels)
    out = model.forward(images)
    #error = out - labels
    #loss = (error*error).mean() #square error, take mean
    print(images.shape)
    print(labels.shape)
    print(out.shape)
    loss = sparse_categorical_crossentropy(out, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.numpy())
    del loss

exit()


idx = 1
Tensor.training = True
for image,label in zip(images,labels):
    out = model.forward(Tensor(image))
    error = out - Tensor(labelVec(label))
    loss += (error*error).mean()
    if math.isnan(loss.data):
        exit()
    if idx % batch_size == 0:
        print(f"batch: {idx}")
        loss = loss/batch_size
        print(f"current loss: {loss.numpy()/batch_size}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del loss
    idx += 1
    #if idx % (batch_size*3) == 0:
    #    exit()


