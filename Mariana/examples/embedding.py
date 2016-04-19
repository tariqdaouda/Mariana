import numpy

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.settings as MSET

MSET.VERBOSE = False

#The first 3 and the last 3 should end up diametrically opposed
data = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
targets = [0, 0, 0, 1, 1, 1]

ls = MS.GradientDescent(lr = 0.5)
cost = MC.NegativeLogLikelihood()

emb = ML.Embedding( 2, nbDimentions = 2, dictSize = len(data), learningScenario = ls, name="emb")
o = ML.SoftmaxClassifier(2, learningScenario = MS.Fixed(), costObject = cost, name = "out")
net = emb > o

miniBatchSize = 2
print "before:"
net.init()

print emb.getEmbeddings()

for i in xrange(2000) :
	for i in xrange(0, len(data), miniBatchSize) :
		net.train(o, emb=data[i:i+miniBatchSize], targets=targets[i:i+miniBatchSize])

print "after:"
print emb.getEmbeddings()
