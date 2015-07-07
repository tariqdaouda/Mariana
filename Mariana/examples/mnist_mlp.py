import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.training.trainers as MT
import Mariana.training.datasetmaps as MDM
import Mariana.training.stopcriteria as MSTOP

from Mariana.examples.useful import load_mnist

"""
This is the equivalent the theano MLP from here: http://deeplearning.net/tutorial/mlp.html
But Mariana style.

This version uses a trainer/dataset mapper setup:
* automatically saves the best model for each set (train, test, validation)
* automatically saves the model if the training halts because of an error or if the process is killed
* saves a log if the process dies unexpectedly
* training results and hyper parameters values are recorded to a file
* allows you to define custom stop criteria
* training info is printed at each epoch, including best scores and at which epoch they were achieved

"""

if __name__ == "__main__":

	# Let's define the network
	ls = MS.GradientDescent(lr=0.01)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(28 * 28, name='inp')
	h = ML.Hidden(500, activation=MA.tanh, decorators=[MD.GlorotTanhInit()], regularizations=[MR.L1(0), MR.L2(0.0001)], name="hid")
	o = ML.SoftmaxClassifier(10, learningScenario=ls, costObject=cost, name="out", regularizations=[MR.L1(0), MR.L2(0.0001)])

	mlp = i > h > o

	mlp.saveDOT("mnist_mlp")
	# And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	trainData = MDM.Series(images=train_set[0], numbers=train_set[1])
	trainMaps = MDM.DatasetMapper()
	trainMaps.map(i, trainData.images)
	trainMaps.map(o, trainData.numbers)

	testData = MDM.Series(images=test_set[0], numbers=test_set[1])
	testMaps = MDM.DatasetMapper()
	testMaps.map(i, testData.images)
	testMaps.map(o, testData.numbers)

	validationData = MDM.Series(images=validation_set[0], numbers=validation_set[1])
	validationMaps = MDM.DatasetMapper()
	validationMaps.map(i, validationData.images)
	validationMaps.map(o, validationData.numbers)

	earlyStop = MSTOP.GeometricEarlyStopping(testMaps, patience=100, patienceIncreaseFactor=1.1, significantImprovement=0.00001, outputLayer=o)
	epochWall = MSTOP.EpochWall(1000)

	trainer = MT.DefaultTrainer(
	    trainMaps=trainMaps,
	    testMaps=testMaps,
	    validationMaps=validationMaps,
	    stopCriteria=[earlyStop, epochWall],
	    trainMiniBatchSize=20
	)

	trainer.start("MLP", mlp)
