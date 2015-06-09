import cPickle
import gzip

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

if __name__ == "__main__" :
	
	#Let's define the network
	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(28*28, name = 'inp')
	h = ML.Hidden(500, activation = MA.tanh, decorators = [MD.GlorotTanhInit()], regularizations = [ MR.L1(0), MR.L2(0.0001) ], name = "hid" )
	o = ML.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out", regularizations = [ MR.L1(0), MR.L2(0.0001) ] )

	mlp = i > h > o
	
	mlp.saveDOT("minist_mlp")
	#And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	trainData = MDM.SynchronizedLists(images = train_set[0], numbers = train_set[1])
	testData = MDM.SynchronizedLists(images = test_set[0], numbers = test_set[1])
	validationData = MDM.SynchronizedLists(images = validation_set[0], numbers = validation_set[1])

	trainMaps = MDM.DatasetMapper()
	trainMaps.map(i, trainData.images)
	trainMaps.map(o, trainData.numbers)

	testMaps = MDM.DatasetMapper()
	testMaps.map(i, testData.images)
	testMaps.map(o, testData.numbers)

	validationMaps = MDM.DatasetMapper()
	validationMaps.map(i, validationData.images)
	validationMaps.map(o, validationData.numbers)

	earlyStop = MSTOP.GeometricEarlyStopping(testMaps, patience = 100, patienceIncreaseFactor = 1.1, significantImprovement = 0.00001, outputLayer = o)
	epochWall = MSTOP.EpochWall(1000)

	trainer = MT.DefaultTrainer(
		trainMaps = trainMaps,
		testMaps = testMaps,
		validationMaps = validationMaps,
		stopCriteria = [earlyStop, epochWall],
		trainMiniBatchSize = 1
	)
	
	trainer.start("MLP", mlp, shuffleMinibatches = False)
