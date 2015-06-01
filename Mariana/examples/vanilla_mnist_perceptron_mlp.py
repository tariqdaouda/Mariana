import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.training.trainers as MT
import Mariana.training.datasetmaps as MDM
import Mariana.training.stopcriteria as MSTOP

"""
This is the equivalent the theano MLP from here: http://deeplearning.net/tutorial/mlp.html
But using Mariana
"""

def load_mnist() :
	"""If i can't find it i will attempt to download it from LISA's place"""
	import urllib, os, gzip, cPickle
	dataset = 'mnist.pkl.gz'
	if (not os.path.isfile(dataset)):
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print '==> Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	f = gzip.open(dataset, 'rb')
	return cPickle.load(f)

def Perceptron() :
	i = ML.Input(28*28, name = 'inp')
	o = ML.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0) ] )

	return i > o

def MLP() :
	import theano

	i = ML.Input(28*28, name = 'inp')
	h = ML.Hidden(500, activation = MA.tanh, decorators = [MD.GlorotTanhInit()], regularizations = [ MR.L1(0), MR.L2(0) ], name = "hid" )
	o = ML.SoftmaxClassifier(10, decorators = [MD.ZerosInit()], learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0.000) ] )

	mlp = i > h > o
	mlp.init()
	
	return mlp
	
if __name__ == "__main__" :
	
	#Let's define the network
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	train_set, validation_set, validation_set = load_mnist()

	model = MLP()
	o = model.outputs.values()[0]

	h = model.layers["hid"]

	maxEpochs = 1000
	miniBatchSize = 20
	
	e = 0
	bestValScore = numpy.inf
	while True :
		trainScores = []
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			res = model.train(o, inp = train_set[0][i : i +20], target = train_set[1][i : i +20] )
			trainScores.append(res[0])
	
		trainScore = numpy.mean(trainScores)
		res = model.test(o, inp = validation_set[0], target = validation_set[1] )
		
		print "---\nepoch", e
		print "\ttrain score:", trainScore*100
		if bestValScore > res[0] :
			bestValScore = res[0]
			print "\tvalidation score:", res[0]*100, "+best+"
		else :
			print "\tvalidation score:", res[0], "best:", bestValScore*100
		
		e += 1