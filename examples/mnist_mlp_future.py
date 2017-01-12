import Mariana.activations as MA
import Mariana.initializations as MI
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

import Mariana.training.future.trainers as MT
import Mariana.training.future.writers as MW
import Mariana.training.future.loggers as MLOG
import Mariana.training.future.datasetmaps as MDM
import Mariana.training.future.stopcriteria as MSTOP

from useful import load_mnist

import Mariana.settings as MSET

MSET.VERBOSE = False

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
	h = ML.Hidden(500, activation=MA.Tanh(), initializations=[MI.GlorotTanhInit(), MI.ZeroBias()], regularizations=[MR.L1(0), MR.L2(0.0001)], name="hid")
	o = ML.SoftmaxClassifier(10, learningScenario=ls, costObject=cost, name="out", regularizations=[MR.L1(0), MR.L2(0.0001)])

	mlp = i > h > o

	mlp.saveDOT("mnist_mlp")
	mlp.saveHTML("mnist_mlp")
	# And then map sets to the inputs and outputs of our network
	train_set, validation_set, test_set = load_mnist()

	trainData = MDM.Series(images=train_set[0], numbers=train_set[1])
	trainMaps = MDM.DatasetMapper("train", miniBatchSize=500)
	trainMaps.mapInput(i, trainData.images)
	trainMaps.mapOutput(o, trainData.numbers)

	testData = MDM.Series(images=test_set[0], numbers=test_set[1])
	testMaps = MDM.DatasetMapper("testAndAccuracy")
	testMaps.mapInput(i, testData.images)
	testMaps.mapOutput(o, testData.numbers)

	validationData = MDM.Series(images=validation_set[0], numbers=validation_set[1])
	validationMaps = MDM.DatasetMapper("testAndAccuracy")
	validationMaps.mapInput(i, validationData.images)
	validationMaps.mapOutput(o, validationData.numbers)

	# earlyStop = MSTOP.GeometricEarlyStopping(testMaps, patience=100, patienceIncreaseFactor=1.1, significantImprovement=0.00001, outputFunction="score", outputLayer=o)
	# epochWall = MSTOP.EpochWall(1000)

	
	ggplot = MW.CSV("MLP", loggers = [MLOG.ParameterMean(), MLOG.ParameterMin(), MLOG.ParameterMax(), MLOG.AbstractionHyperParameters()], writeRate=1)
	pp = MW.PrettyPrinter(loggers = [MLOG.ParameterMean(paramList = ["hid.W"] ), MLOG.ParameterMax(paramList=["hid.W"]), MLOG.AbstractionHyperParameters(layerNames=["hid"]), MLOG.Scores()])

	trainer = MT.DefaultTrainer(
		trainMaps=trainMaps,
		testMaps=testMaps,
		validationMaps=validationMaps,
		model=mlp,
		onEpochEnd=[ggplot, pp]
		# stopCriteria=[earlyStop, epochWall],
		# testFunctionName="testAndAccuracy",
		# validationFunctionName="testAndAccuracy",
		# trainMiniBatchSize=20,
		# saveIfMurdered=False
	)

	# recorder = MREC.GGPlot2("MLP", whenToSave = [MREC.SaveMin("test", o.name, "score")], printRate=1, writeRate=1)
	trainer.start("MLP")
