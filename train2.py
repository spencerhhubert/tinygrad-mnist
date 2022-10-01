from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from extra.training import train, evaluate
from extra.utils import get_parameters
from datasets import fetch_mnist
import random

# load the mnist dataset
X_train, Y_train, X_test, Y_test = fetch_mnist()

# create a model
class TinyBobNet:
  def __init__(self):
    self.l1 = Tensor.uniform(784, 128)
    self.l2 = Tensor.uniform(128, 10)

  def parameters(self):
    return get_parameters(self)

  def forward(self, x):
    return x.dot(self.l1).relu().dot(self.l2).logsoftmax()

model = TinyBobNet()
optimizer = optim.SGD(model.parameters(), lr=0.001)
train(model, X_train, Y_train, optimizer, BS=69, steps=1000)

from mnist import MNIST
mndata = MNIST("data")
mndata.gz = True
data = mndata.load_testing()

def getRandomData():
    val = int(random.random()*len(data[0]))
    image, label = data[0][val], data[1][val]
    return (image, label)

def getPrediction(outs:list):
    outs = outs.data
    values = list(map(lambda x : x, outs))
    high = values[0]
    high_idx = 0
    for val,i in zip(values,range(len(values)-1)):
        if val > high:
            high = val
            high_idx = i
    return range(len(outs))[high_idx]
for i in range(10):
    image, label = getRandomData()
    print(f"Actual: {label}")
    print(mndata.display(image))
    out = model.forward(Tensor(image))
    print(out)
    print(f"Prediction: {getPrediction(out)}")
