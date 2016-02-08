import numpy, theano

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

####
# Class 1 inputs have a pattern of ones on the left side, class 2 inputs have it on the right.
# This network learn to differentiate between the two
####

class ConvWithChanneler :
	
	def __init__(self, inputSize, patternSize, ls, cost) :
		# pooler = MCONV.NoPooling()
		pooler = MCONV.MaxPooling2D(1, 2)
		
		#The input channeler will take regular layers and arrange them into several channels
		i = ML.Input(inputSize, name = 'inp')
		ichan = MCONV.InputChanneler(1, inputSize, name = 'inpChan')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 5,
			filterHeight = 1,
			filterWidth = patternSize/2,
			activation = MA.ReLU(),
			pooler = pooler,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 10,
			filterHeight = 1,
			filterWidth = patternSize/2,
			activation = MA.ReLU(),
			pooler = MCONV.NoPooling(),
			name = "conv2"
		)

		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(5, activation = MA.ReLU(), decorators = [], regularizations = [], name = "hid" )
		o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [] )
		
		self.model = i > ichan > c1 > c2 > f > h > o
		
	def train(self, inputs, targets) :
		return self.model.train("out", inp = inputs, targets = targets )

	def test(self, inputs, targets) :
		return self.model.test("out", inp = inputs, targets = targets )

def makeDataset(nbExamples, size, patternSize, testRatio = 0.2, easy = True) :
	data = numpy.random.randn(nbExamples, size).astype(theano.config.floatX)
	pattern = numpy.ones(patternSize)
	
	targets = []
	for i in xrange(len(data)) :
		if i%2 == 0 :
			if not easy :
				start = numpy.random.randint(0, size/2 - patternSize)
			else :
				start = 1
			targets.append(0)
		else :
			if not easy :
				start = numpy.random.randint(size/2, size - patternSize)
			else :
				start = size - patternSize -1
			targets.append(1)

		data[i][start:start+patternSize] = pattern

	targets = numpy.asarray(targets, dtype=theano.config.floatX)
	
	lenTest = len(data) * 0.2
	trainData, trainTargets = data[lenTest:], targets[lenTest:]
	testData, testTargets = data[:lenTest], targets[:lenTest]

	return ( (trainData, trainTargets), (testData, testTargets ) )

if __name__ == "__main__" :
	
	ls = MS.GradientDescent(lr = 0.5)
	cost = MC.NegativeLogLikelihood()

	maxEpochs = 2000
	miniBatchSize = 256
	inputSize = 128
	patternSize = 5

	model = ConvWithChanneler(inputSize, patternSize, ls, cost)
	
	trainSet, testSet = makeDataset(10000, inputSize, patternSize, easy = False)

	bestValScore = numpy.inf
	
	printRate = 10
	for epoch in xrange(maxEpochs) :
		trainScores = []
		for i in xrange(0, len(trainSet[0]), miniBatchSize) :
			inputs = trainSet[0][i : i +miniBatchSize]
			targets = trainSet[1][i : i +miniBatchSize]
			res = model.train(inputs, targets )
			trainScores.append(res[0])
		
		if epoch%printRate == 0 :
			trainScore = numpy.mean(trainScores)
			res = model.test(testSet[0], testSet[1] )
			
			print "---\nepoch", epoch
			print "\ttrain score:", trainScore
			if bestValScore > res[0] :
				bestValScore = res[0]
				print "\tvalidation score:", res[0], "+best+"
			else :
				print "\tvalidation score:", res[0], "best:", bestValScore
