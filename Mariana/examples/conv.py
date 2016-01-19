import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

from Mariana.examples.useful import load_mnist

def conv(ls, cost) :
	maxPool = ML.MaxPooling2D(2, 2)
	passPool = ML.Pass()
	
	i = ML.Input(28*28, name = 'inp')
	c1 = ML.Convolution2D( 
		nbMaps= 5,
		imageHeight = 28,
		imageWidth = 28,
		filterHeight = 12,
		filterWidth = 1,
		activation = MA.Tanh(),
		pooler = maxPool,
		# pooler = passPool,
		name = "conv"
	)

	# c2 = ML.Convolution2D( 
	# 	nbMaps= 5,
	# 	imageHeight = 28,
	# 	imageWidth = 28,
	# 	filterHeight = 12,
	# 	filterWidth = 1,
	# 	activation = MA.Tanh(),
	# 	pooler = maxPool,
	# 	# pooler = passPool,
	# 	name = "conv"
	# )
	f = ML.Flatten(name = "flat")
	# h = ML.Hidden(500, activation = MA.Tanh(), decorators = [MD.GlorotTanhInit()], regularizations = [ MR.L1(0), MR.L2(0.0001) ], name = "hid" )
	o = ML.SoftmaxClassifier(10, decorators = [], learningScenario = ls, costObject = cost, name = "out", regularizations = [ ] )

	mlp = i > c1 > f > o
	
	return mlp


if __name__ == "__main__" :
	
	#Let's define the network
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	train_set, validation_set, validation_set = load_mnist()

	maxEpochs = 1000
	miniBatchSize = 10
	
	model = conv(ls, cost)
	o = model.outputs.values()[0]

	epoch = 0
	bestValScore = numpy.inf
	model.init()
	
	while True :
		trainScores = []
		for i in xrange(0, len(train_set[0]), miniBatchSize) :
			inputs = train_set[0][i : i +miniBatchSize]
			#print inputs
			# print model["conv"].convolution.eval({ model["inp"].inputs : inputs } ).shape
			# print "----a"
			# print model["conv"].pooled.eval({ model["inp"].inputs : inputs } ).shape
			# print "----b"
			# print model["conv"].outputs.eval({ model["inp"].inputs : inputs } ).shape
			# print "----c"
			# print model["flat"].outputs.eval({ model["inp"].inputs : inputs } )
			# print model["flat"].outputs.eval({ model["inp"].inputs : inputs } ).shape
			
			# print "----d"
			# print train_set[1][i : i +miniBatchSize]
			# print i, len(train_set[0])
			res = model.train(o, inp = inputs, targets = train_set[1][i : i +miniBatchSize] )
			trainScores.append(res[0])
	
		trainScore = numpy.mean(trainScores)
		# res = model.test(o, inp = validation_set[0], targets = validation_set[1] )
		
		print "---\nepoch", epoch
		print "\ttrain score:", trainScore
		# if bestValScore > res[0] :
			# bestValScore = res[0]
			# print "\tvalidation score:", res[0], "+best+"
		# else :
			# print "\tvalidation score:", res[0], "best:", bestValScore
		
		epoch += 1
