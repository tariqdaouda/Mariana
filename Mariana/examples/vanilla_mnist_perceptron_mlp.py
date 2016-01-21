import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

from Mariana.examples.useful import load_mnist

"""
This is the equivalent the theano MLP from here: http://deeplearning.net/tutorial/mlp.html
But Mariana style.

The vanilla version does not use the a trainer or a dataset mapper. It should therefor be a bit faster,
but you don't get all the niceties provided by the trainer.
"""

def Perceptron(ls, cost) :
	i = ML.Input(28*28, name = 'inp')
	o = ML.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0) ] )

	return i > o

def MLP(ls, cost) :
	
	i = ML.Input(28*28, name = 'inp')
	h = ML.Hidden(500, activation = MA.Tanh(), decorators = [MD.GlorotTanhInit()], regularizations = [ MR.L1(0), MR.L2(0.0001) ], name = "hid" )
	o = ML.SoftmaxClassifier(10, decorators = [MD.ZerosInit()], learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0.0001) ] )

	mlp = i > h > o
	
	return mlp

if __name__ == "__main__" :
	
	#Let's define the network
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	train_set, validation_set, validation_set = load_mnist()

	maxEpochs = 1000
	miniBatchSize = 20
	
	model = MLP(ls, cost)
	o = model.outputs.values()[0]
	
	epoch = 0
	bestValScore = numpy.inf
	
	while True :
		trainScores = []
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			#you can also use the name of the output layer as defined by its attribute 'name': 
			#res = model.train("out", ... )
			res = model.train(o, inp = train_set[0][i : i +miniBatchSize], targets = train_set[1][i : i +miniBatchSize] )
			trainScores.append(res[0])
	
		trainScore = numpy.mean(trainScores)
		res = model.test(o, inp = validation_set[0], targets = validation_set[1] )
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		if bestValScore > res[0] :
			bestValScore = res[0]
			print "\tvalidation score:", res[0], "+best+"
		else :
			print "\tvalidation score:", res[0], "best:", bestValScore
		
		epoch += 1