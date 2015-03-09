import cPickle
import gzip

from Mariana.layers import *
from Mariana.rules import *
from Mariana.trainers import *
import theano.tensor as tt

miniBatchSize = 20
nbEpochs = -1

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, validation_set, test_set = cPickle.load(f)
train_set_targets = train_set[1].astype('int32')
test_set_targets = test_set[1].astype('int32')
validation_set_targets = validation_set[1].astype('int32')

dataset = {}
dataset["train"] = {"examples" : {"images" : train_set[0]}, "targets" : {"class" : train_set_targets}}
dataset["test"] = {"examples" : {"images" : test_set[0]}, "targets" : {"class" : test_set_targets}}
dataset["validation"] = {"examples" : {"images" : validation_set[0]}, "targets" : {"class" : validation_set_targets}}

ls = DefaultScenario(lr = 0.01, momentum = 0)
cost = NegativeLogLikelihood(l1 = 0, l2 = 0.0001)

i = Input(28*28, 'inp')
h = Hidden(500, activation = tt.tanh)
o = SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out")

mlp = i > h > o

inputMaps = Mapper()
inputMaps.add("inp", "images")
outputMaps = Mapper()
outputMaps.add("out", "class")

trainer = NoEarlyStopping()
trainer.run("MLP", mlp, dataset, nbEpochs = nbEpochs, miniBatchSize = miniBatchSize, inputMaps = inputMaps, outputMaps = outputMaps)