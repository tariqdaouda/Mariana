import cPickle
import gzip

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.trainers as MT

"""
This is the equivalent the theano MLP from here: http://deeplearning.net/tutorial/mlp.html
But using Mariana
"""

def load_mnist() :
	"""If i can't find it i will attempt to download it from lisa's place"""
	import urllib, os
	dataset = 'mnist.pkl.gz'
	if (not os.path.isfile(dataset)):
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print '==> Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	f = gzip.open(dataset, 'rb')
	return cPickle.load(f)

if __name__ == "__main__" :
	
	miniBatchSize = 20
	nbEpochs = -1

	#Let's define the network
	ls = MS.DefaultScenario(lr = 0.01, momentum = 0)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(28*28, 'inp')
	h = ML.Hidden(500, activation = MA.tanh, decorators = [MD.GlorotTanhInit()], regularizations = [ MR.L1(0), MR.L2(0.0001) ] )
	o = ML.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0.0001) ] )

	mlp = i > h > o

	#And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	trainMaps = MT.DatasetMapper()
	trainMaps.addInput("inp", train_set[0])
	trainMaps.addOutput("out", train_set[1].astype('int32'))

	testMaps = MT.DatasetMapper()
	testMaps.addInput("inp", test_set[0])
	testMaps.addOutput("out", test_set[1].astype('int32'))

	validationMaps = MT.DatasetMapper()
	validationMaps.addInput("inp", validation_set[0])
	validationMaps.addOutput("out", validation_set[1].astype('int32'))

	trainer = MT.Trainer()
	trainer.start("MLP", mlp, trainMaps = trainMaps, testMaps = testMaps, validationMaps = validationMaps, stopCriteria = [], miniBatchSize = miniBatchSize)