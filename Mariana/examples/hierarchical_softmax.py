import Mariana.layers as ML
import Mariana.initializations as MI
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.activations as MA
import theano, numpy

import Mariana.settings as MSET
MSET.VERBOSE = False

##
## This example implements a two level softmax and learns the following conditional tree
##		 root
##		/    \
##		C     R => first softmax
##     / \
##	  C1  C2    => second softmax
##

def makeDataset(nbExamples, size, patternSize) :
	data = numpy.random.randn(nbExamples, size).astype(theano.config.floatX)
	patternC1 = numpy.ones(patternSize)
	
	patternC2 = numpy.ones(patternSize)
	patternC2[0] = 0
	patternC2[-1] = 0

	targets = []
	for i in xrange(len(data)) :
		start = 1
		if i%2 == 0 :
			data[i][start:start+patternSize] = patternC1
			targets.append(1)
		elif i%2 == 0 :
			data[i][start:start+patternSize] = patternC2
			targets.append(2)
		else :
			targets.append(0)

	targets = numpy.asarray(targets, dtype=theano.config.floatX)
	return data, targets

if __name__ == "__main__" :
	examples, targets = makeDataset(300, 100, 10)

	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(100, 'inp')
	h1 = ML.Hidden(50, activations = MA.ReLU())
	h2 = ML.Hidden(2, activations = MA.Softmax())
	o = ML.SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")

	mlp = i > h1 > h2 > o

	for k in xrange(1000) :
		for example, target in zip(examples, targets) :
			mlp.train(o, inp=[example], targets=[target])

	nbErr = 0
	for example, target in zip(examples, targets) :
		if target != mlp.classify(o, inp=[example])[0] :
			nbErr += 1

	print "Nb Errors: %s/%s (%s%%) " % (nbErr, len(targets), float(nbErr)/len(targets) * 100)