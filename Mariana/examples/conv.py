import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.settings as MSET

MSET.VERBOSE = False

from Mariana.examples.useful import load_mnist

def conv(ls, cost) :
	maxPool = MCONV.MaxPooling2D(2, 2)
	passPool = MCONV.Pass()
	
	i = MCONV.Input(1, 28, 28, name = 'inp')
	c1 = MCONV.Convolution2D( 
		nbChannels = 1,
		filterHeight = 5,
		filterWidth = 5,
		activation = MA.Tanh(),
		pooler = maxPool,
		name = "conv1"
	)

	c2 = MCONV.Convolution2D( 
		nbChannels = 1,
		filterHeight = 5,
		filterWidth = 5,
		activation = MA.Tanh(),
		pooler = maxPool,
		name = "conv2"
	)

	f = MCONV.Flatten(name = "flat")
	o = ML.SoftmaxClassifier(10, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )

	conv = i > c1 > c2 > f > o
	
	return conv

def conv2(ls, cost) :
	maxPool = MCONV.MaxPooling2D(2, 2)
	passPool = MCONV.Pass()
	
	i = ML.Input(28*28, name = 'inp')
	ichan = MCONV.InputChanneler(28, 28, name = 'inpChan')
	
	c1 = MCONV.Convolution2D( 
		nbChannels = 1,
		filterHeight = 5,
		filterWidth = 5,
		activation = MA.Tanh(),
		pooler = maxPool,
		# pooler = passPool,
		name = "conv1"
	)

	c2 = MCONV.Convolution2D( 
		nbChannels = 1,
		filterHeight = 5,
		filterWidth = 5,
		activation = MA.Tanh(),
		pooler = maxPool,
		# pooler = passPool,
		name = "conv2"
	)

	f = MCONV.Flatten(name = "flat")
	o = ML.SoftmaxClassifier(10, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )

	conv = i > ichan > c1 > c2 > f > o
	
	return conv

if __name__ == "__main__" :
	
	#Let's define the network
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	train_set, validation_set, validation_set = load_mnist()

	maxEpochs = 1000
	miniBatchSize = 10
	
	model = conv2(ls, cost)
	o = model.outputs.values()[0]

	epoch = 0
	bestValScore = numpy.inf
	model.init()
	
	while True :
		trainScores = []
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			# inputs = train_set[0][i : i +miniBatchSize].reshape((-1, 1, 28, 28))
			inputs = train_set[0][i : i +miniBatchSize]
			res = model.train(o, inp = inputs, targets = train_set[1][i : i +miniBatchSize] )
			trainScores.append(res[0])
	
		trainScore = numpy.mean(trainScores)
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		
		epoch += 1
