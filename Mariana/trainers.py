import sys, numpy, cPickle, time

from pyGeno.tools.parsers.CSVTools import CSVFile

from Mariana.layers import *
import theano.tensor as tt

class Mapper(object):
 	"""This class is here to map for example: inputs in the network to data in the dataset, or outputs to targets """
 	def __init__(self):
 		self.maps = {}

 	def add(self, layerName, theSet) :
 		self.maps[layerName] = theSet

 	def __str__(self) :
 		return str(self.maps).replace(" : ", " => ")

 	def __getattr__(self, k) :
 		return getattr(object.__getattribute__(self, "maps"), k)

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, cliffEpochs = 0, significantImprovement = 0.00001):
		self.significantImprovement = significantImprovement
		self.cliffEpochs = cliffEpochs
		self.bestValidationErr = numpy.inf

	def run(self, model, dataset) :
		raise NotImplemented("Must be implemented in child")

class NoEarlyStopping(Trainer) :

	def run(self, name, model, dataset, nbEpochs, miniBatchSize, inputMaps, outputMaps) :

		def _parseInputs(inputMaps, examples, ind, miniBatchSize) :
			res = {}
			for inputName, setName in inputMaps.iteritems() :
				res[inputName] =  examples[setName][ind: ind + miniBatchSize]
			return res

		csvEvolution = CSVFile(["epoch", "type", "score"])
		csvEvolution.streamToFile("%s-evolution.csv" % name)

		trainSet = dataset["train"]
		testSet = dataset["test"]
		validationSet = dataset["validation"]

		print "learning..."
		evolution = {"train" : [], "validation" : [], 'test' : []}
		firstKey = trainSet["examples"].keys()[0]
		iop = 0
		while (nbEpochs < 0) or (iop < nbEpochs):
			meanTrain = []
			meanVal = []
			meanTest = []
			for i in xrange(0, len(trainSet["examples"][firstKey]), miniBatchSize) :
				kwargs = _parseInputs(inputMaps, trainSet["examples"], i, miniBatchSize)
				for outputName, targetName in outputMaps.iteritems() :
					kwargs.update({ "target" : trainSet["targets"][targetName][i: i + miniBatchSize] })
					res = model.train(outputName, **kwargs)
					meanTrain.append(res[0])

			kwargs = _parseInputs(inputMaps, validationSet["examples"], 0, len(validationSet["examples"][firstKey]))
			for outputName, targetName in outputMaps.iteritems() :
				kwargs.update({ "target" : validationSet["targets"][targetName] } )
				res = model.test(outputName, **kwargs)
				meanVal.append(res[0])

			kwargs = _parseInputs(inputMaps, testSet["examples"], 0, len(testSet["examples"][firstKey]))
			for outputName, targetName in outputMaps.iteritems() :
				kwargs.update({ "target" : testSet["targets"][targetName] } )
				res = model.test(outputName, **kwargs)
				meanTest.append(res[0])


			if nbEpochs > 0 :
				print "epoch %s/%s (%.2f%%), mean train err %s, val err %s, test err %s" %(iop, nbEpochs, float(iop)/nbEpochs*100, mt, mv, mte)
			else :
				print "epoch %s/%s, mean train err %s, val err %s, test err %s" %(iop, nbEpochs, mt, mv, mte)
			
			mt, mv, mte = numpy.mean(meanTrain), numpy.mean(meanVal), numpy.mean(meanTest)
			if (mv - self.bestValidationErr) < self.significantImprovement :
				filename = "%s-best_valerr" % (name)
				print "\t>%s: new best validation score %s -> %s" % (name, self.bestValidationErr, mv)
				self.bestValidationErr = mv
				if iop > self.cliffEpochs :
					print "saving model to %s..." % (filename)
					model.save(filename)
					f = open(filename + ".txt", 'w')
					f.write("date: %s\nbest validation err: %s\ntest err:%s\ntrain err:%s" % (time.ctime(), mv, mte, mt))
					f.close()
			line = csvEvolution.newLine()
			line["epoch"] = iop
			line["type"] = "Train"
			line["score"] = mt
			line.commit()
			line = csvEvolution.newLine()
			line["epoch"] = iop
			line["type"] = "Test"
			line["score"] = mte
			line.commit()
			line = csvEvolution.newLine()
			line["epoch"] = iop
			line["type"] = "Validation"
			line["score"] = mv
			line.commit()

			sys.stdout.flush()
			iop += 1