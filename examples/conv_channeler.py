import numpy

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

from useful import load_mnist

####
# This is an example of two identical conv nets. The only difference is that the first one uses a convolution input layer
# while the second uses an InpuChanneler to automatically cast the output of a regular input layer in a format
# suitable for a convolution layer.

class Conv :
	
	def __init__(self, ls, cost) :
		maxPool = MCONV.MaxPooling2D(2, 2)
		
		i = MCONV.Input(nbChannels = 1, height = 28, width = 28, name = 'inp')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 20,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 50,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv2"
		)

		#needed for the transition to a fully connected layer
		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(500, activation = MA.Tanh(), decorators = [], regularizations = [ ], name = "hid" )
		o = ML.SoftmaxClassifier(10, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )
		
		self.model = i > c1 > c2 > f > h > o
		print self.model
		
	def train(self, inputs, targets) :
		#The inputs have to be reshaped into a 4d matrix before passing them to the conv layers
		#Because of that it is MUCH slower than ConvWithChanneler
		inps = inputs.reshape((-1, 1, 28, 28))
		return self.model.train("out", inp = inps, targets = targets)

class ConvWithChanneler :
	
	def __init__(self, ls, cost) :
		maxPool = MCONV.MaxPooling2D(2, 2)
		
		#The input channeler will take regular layers and arrange them into several channels
		i = ML.Input(28*28, name = 'inp')
		ichan = MCONV.InputChanneler(28, 28, name = 'inpChan')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 1,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 1,
			filterHeight = 5,
			filterWidth = 5,
			activation = MA.Tanh(),
			pooler = maxPool,
			name = "conv2"
		)

		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(5, activation = MA.Tanh(), decorators = [], regularizations = [ ], name = "hid" )
		o = ML.SoftmaxClassifier(10, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )
		
		self.model = i > ichan > c1 > c2 > f > h > o

	def train(self, inputs, targets) :
		#because of the channeler there is no need to reshape the data besfore passing them to the conv layer
		return self.model.train("out", inp = inputs, targets = targets )

if __name__ == "__main__" :
	
	ls = MS.GradientDescent(lr = 0.1)
	cost = MC.NegativeLogLikelihood()

	train_set, validation_set, validation_set = load_mnist()

	maxEpochs = 200
	miniBatchSize = 500
	
	model = Conv(ls, cost)
	
	epoch = 0
	while True :
		trainScores = []
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			inputs = train_set[0][i : i +miniBatchSize]
			targets = train_set[1][i : i +miniBatchSize]
			res = model.train(inputs, targets )
			trainScores.append(res["score"])

		trainScore = numpy.mean(trainScores)
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		
		epoch += 1