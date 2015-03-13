import cPickle
import gzip

from Mariana.layers import *
from Mariana.rules import *
from Mariana.trainers import *

"""
This is the equivalent the theano MLP from here: http://deeplearning.net/tutorial/mlp.html
But Mariana style

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
	ls = DefaultScenario(lr = 0.01, momentum = 0)
	cost = NegativeLogLikelihood(l1 = 0, l2 = 0.0001)

	i = Input(28*28, 'inp')
	h = Hidden(500, activation = tt.tanh)
	o = SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "out")

	mlp = i > h > o

	#And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	trainMaps = DatasetMapper()
	trainMaps.addInput("inp", train_set[0])
	trainMaps.addOutput("out", train_set[1].astype('int32'))

	testMaps = DatasetMapper()
	testMaps.addInput("inp", test_set[0])
	testMaps.addOutput("out", test_set[1].astype('int32'))

	#and train
	trainer = NoEarlyStopping()
	trainer.run("MLP", mlp, trainMaps = trainMaps, testMaps = testMaps, nbEpochs = nbEpochs, miniBatchSize = miniBatchSize)