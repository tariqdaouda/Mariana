import cPickle, time, sys, traceback
import numpy

from pyGeno.tools.parsers.CSVTools import CSVFile

from Mariana.layers import *
import theano.tensor as tt

class DatasetMapper(object):
 	"""This class is here to map inputs of a network to datasets, or outputs to target sets.
 	If forceSameLengths all sets must have the same length."""
 	def __init__(self, forceSameLengths = True):
 		self.inputs = {}
 		self.outputs = {}
 		self.forceSameLengths = forceSameLengths
 		self.length = 0

 	def _add(self, dct, layerName, aSet) :
 		"""This function is here because i don't like repeating myself"""
 		if self.forceSameLengths and self.length != 0 and len(aSet) != self.length:
 			raise ValueError("All sets must have the same number of elements. len(aSet) = %s, but another set has a length of %s" % (len(aSet), self.length))

 		if layerName in self.inputs or layerName in self.outputs:
 			raise ValueError("There is already a mapped layer by that name")
 		
 		if len(aSet) > self.length :
	 		self.length = len(aSet)
 		dct[layerName] = aSet

 	def addInput(self, layerName, aSet) :
 		"""Adds a mapping rule ex: .add("input1", dataset["train"]) """
 		self._add(self.inputs, layerName, aSet)

 	def addOutput(self, layerName, aSet) :
 		"""Adds a mapping rule ex: .add("output1", dataset["train"]) """
 		self._add(self.outputs, layerName, aSet)

 	def _getLayerBatch(self, dct, layerName, i, size) :
 		"""This function is here because i don't like repeating myself"""
 		if size == "all" :
 			ssize = len(dct[layerName])
 		else :
 			ssize = size

 		if self.forceSameLengths :
	 		ii = i % len(dct[layerName])
 		else :
 			ii = i

 		return dct[layerName][ii:ii+ssize]

 	def _getBatches(self, dct, i, size) :
 		"""This function is here because i don't like repeating myself"""
 		res = {}
 		for k in dct :
 			res[k] = self._getLayerBatch(dct, k, i, size)

 		return res

	def getInputLayerBatch(self, layerName, i, size) :
		"""Returns a batch for a given input layer sarting at position i, and of length size.
 		If i is supperior to the length of the set and self.forceSameLengths is false,
 		them i will become i % len(set). If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
 		return self._getLayerBatch(self.inputs, layerName, i, size)

 	def getInputBatches(self, i, size) :
 		"""Applies getInputLayerBatch iteratively for all layers and returns a dict:
 		layer name => batch"""
 		return self._getBatches(self.inputs, i, size)

 	def getOutputLayerBatch(self, layerName, i, size) :
		"""Returns a batch for a given output layer sarting at position i, and of length size.
 		If i is supperior to the length of the set and self.forceSameLengths is false,
 		them i will become i % len(set). If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
 		return self._getLayerBatch(self.outputs, layerName, i, size)
 	
 	def getOutputBatches(self, i, size) :
		"""Applies getOutputLayerBatch iteratively for all layers and returns a dict:
 		layer name => batch"""
 		return self._getBatches(self.outputs, i, size)

 	def __len__(self) :
 		return self.length

 	def __str__(self) :
 		return str(self.inputs).replace(" : ", " => ")  + "\n" + str(self.outputs).replace(" : ", " => ") 

class Trainer(object):
	"""All Trainers must inherit from me"""
	def __init__(self, cliffEpochs = 0, saveOnException = True, significantImprovement = 0.00001):
		self.significantImprovement = significantImprovement
		self.cliffEpochs = cliffEpochs
		self.saveOnException = saveOnException
		self.bestValidationErr = numpy.inf
		self.bestTestErr = numpy.inf

	def start(self, name, model, *args, **kwargs) :
		"""Starts the training. If anything bad and unespected happens during training, the Trainer
		will attempt to save the model and logs."""

		def _dieGracefully() :
			exType, ex, tb = sys.exc_info()
			# traceback.print_tb(tb)
			death_time = time.ctime().replace(' ', '_')
			filename = "dx-xb_" + name + "_death_by_" + exType.__name__ + "_" + death_time
			sys.stderr.write("\n===\nDying gracefully from %s, and saving myself to:\n...%s\n===\n" % (exType, filename))
			model.save(filename)
			f = open(filename +  ".traceback.log", 'w')
			f.write("Mariana training Interruption\n=============================\n")
			f.write("\nDetails\n-------\n")
			f.write("Name: %s\n" % name)
			f.write("Killed by: %s\n" % str(exType))
			f.write("Time of death: %s\n" % death_time)
			f.write("Model saved to: %s\n" % filename)
			f.write("\nTraceback\n---------\n")

			f.write(str(traceback.extract_tb(tb)).replace("), (", "),\n(").replace("[(","[\n(").replace(")]",")\n]"))
			f.close()

		if not self.saveOnException :
			return self.run(name, model, *args, **kwargs)
		else :
			try :
				return self.run(name, model, *args, **kwargs)
			except KeyboardInterrupt, Exception :
				_dieGracefully()
				raise

	def run(self, name, model, *args, **kwargs) :
		"""The actual function that is called by start, and the one that must be implemented in children"""
		raise NotImplemented("Must be implemented in child")

class NoEarlyStopping(Trainer) :
	"""This trainner simply never stops until you kill it. It will create a CSV file <model-name>-evolution.csv
	to log the errors for test and train sets for each epoch, and will save the current model each time a better test 
	error is achieved."""

	def run(self, name, model, trainMaps, testMaps, nbEpochs, miniBatchSize) :

		startTime = time.time()
		csvEvolution = CSVFile(["epoch", "runtime", "type", "score"])
		csvEvolution.streamToFile("%s-evolution.csv" % name)

		print "learning..."
		iop = 0
		while (nbEpochs < 0) or (iop < nbEpochs):
			meanTrain = []
			meanTest = []
			for i in xrange(0, len(trainMaps), miniBatchSize) :
				kwargs = trainMaps.getInputBatches(i, miniBatchSize)
				for outputName in trainMaps.outputs.iterkeys() :
					kwargs.update({ "target" : trainMaps.getOutputLayerBatch(outputName, i, miniBatchSize)} )
					res = model.train(outputName, **kwargs)
					meanTrain.append(res[0])

			kwargs = testMaps.getInputBatches(0, size = "all")
			for outputName in testMaps.outputs.iterkeys() :
				kwargs.update({ "target" : testMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
				res = model.test(outputName, **kwargs)
				meanTest.append(res[0])

			mt, mte = numpy.mean(meanTrain), numpy.mean(meanTest)
			runtime = int(time.time() - startTime)
			
			if nbEpochs > 0 :
				print "epoch %s/%s (%.2f%%), mean train err %s, test err %s" %(iop, nbEpochs, float(iop)/nbEpochs*100, mt, mte)
			else :
				print "epoch %s/%s, mean train err %s, test err %s" %(iop, nbEpochs, mt, mte)
			
			if (mte - self.bestTestErr) < self.significantImprovement :
				filename = "%s-best_Testerr" % (name)
				print "\t>%s: new best test score %s -> %s" % (name, self.bestTestErr, mte)
				self.bestTestErr = mte
				if iop > self.cliffEpochs :
					print "saving model to %s..." % (filename)
					model.save(filename)
					f = open(filename + ".txt", 'w')
					f.write("date: %s\n runtime: %s\n epoch: %s\n best test err: %s\ntrain err:%s" % (time.ctime(), runtime, iop, mte, mt))
					f.close()

			line = csvEvolution.newLine()
			line["epoch"] = iop
			line["runtime"] = runtime
			line["type"] = "Train"
			line["score"] = mt
			line.commit()
			line = csvEvolution.newLine()
			line["epoch"] = iop
			line["runtime"] = runtime
			line["type"] = "Test"
			line["score"] = mte
			line.commit()

			sys.stdout.flush()
			iop += 1