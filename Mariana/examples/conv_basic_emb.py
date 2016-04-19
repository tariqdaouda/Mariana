import numpy, theano

import Mariana.activations as MA
import Mariana.layers as ML
import Mariana.convolution as MCONV
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.settings as MSET

####
# Class 1 inputs have a pattern of ones on the left side, class 2 inputs have it on the right.
# This network learn to differentiate between the two
####

class ConvEmb :
	
	def __init__(self, inputSize, dictSize, patternSize, embSize, ls, cost) :
		# pooler = MCONV.NoPooling()
		pooler = MCONV.MaxPooling2D(1, 2)
		
		emb = MCONV.Embedding(size=inputSize, nbDimentions=embSize, dictSize=dictSize, name = 'Emb')
		
		c1 = MCONV.Convolution2D( 
			nbFilters = 1,
			filterHeight = 1,
			filterWidth = patternSize/2,
			activation = MA.ReLU(),
			pooler = pooler,
			name = "conv1"
		)

		c2 = MCONV.Convolution2D( 
			nbFilters = 4,
			filterHeight = 1,
			filterWidth = patternSize/2,
			activation = MA.ReLU(),
			pooler = MCONV.NoPooling(),
			name = "conv2"
		)

		f = MCONV.Flatten(name = "flat")
		h = ML.Hidden(5, activation = MA.ReLU(), decorators = [], regularizations = [], name = "hid" )
		o = ML.SoftmaxClassifier(2, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [] )
		
		self.model = emb > c1 > c2 > f > h > o
		
	def train(self, inputs, targets) :
		return self.model.train("out", Emb = inputs, targets = targets )

	def test(self, inputs, targets) :
		return self.model.test("out", Emb = inputs, targets = targets )

def makeDataset(inputSize, nbExamples, dictSize, patternSize) :

	data = numpy.random.randint(0, dictSize, (nbExamples, inputSize)).astype("int8")
	targets = []
	for i in xrange(len(data)) :
		if i%2 == 0 :
			targets.append(0)
		else :
			targets.append(1)
			r = numpy.random.randint(inputSize - patternSize)
			data[i][r:r+patternSize] = 1

	return data, targets

if __name__ == "__main__" :

	MSET.VERBOSE = False

	ls = MS.GradientDescent(lr = 0.5)
	cost = MC.NegativeLogLikelihood()

	maxEpochs = 2000
	miniBatchSize = 2
	inputSize = 10
	dictSize = 4
	patternSize = 4
	embSize = 2

	data, targets = makeDataset(inputSize, 1000, dictSize, patternSize)

	model = ConvEmb(inputSize, dictSize, patternSize, embSize, ls, cost)
	
	miniBatchSize = 2
	print "before:"
	model.model.init()
	print model.model["Emb"].getEmbeddings()

	for i in xrange(200) :
		for i in xrange(0, len(data), miniBatchSize) :
			d = data[i:i+miniBatchSize]
			model.train(data[i:i+miniBatchSize], targets[i:i+miniBatchSize])

	print "after:"
	print model.model["Emb"].getEmbeddings()
