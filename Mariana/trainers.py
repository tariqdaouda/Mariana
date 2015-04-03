import cPickle, time, sys, traceback, random, types
import numpy
from collections import OrderedDict

from pyGeno.tools.parsers.CSVTools import CSVFile

from Mariana.layers import *
import theano.tensor as tt

class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	pass

class DatasetMapper(object):
 	"""This class is here to map inputs of a network to datasets, or outputs to target sets.
 	If forceSameLengths all sets must have the same length."""
 	def __init__(self):
 		self.inputs = {}
 		self.outputs = {}
 		self.length = 0
 		self.randomIds = None
 		self.locked = False

 	def _add(self, dct, layerName, aSet) :
 		"""This function is here because i don't like repeating myself"""
 		if self.length != 0 and len(aSet) != self.length:
 			raise ValueError("All sets must have the same number of elements. len(aSet) = %s, but another set has a length of %s" % (len(aSet), self.length))

 		if layerName in self.inputs or layerName in self.outputs:
 			raise ValueError("There is already a mapped layer by that name")
 		
 		if len(aSet) > self.length :
	 		self.length = len(aSet)
 		dct[layerName] = aSet

 	def addInput(self, layerName, aSet) :
 		"""Adds a mapping rule ex: .add("input1", dataset["train"])"""
 		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.inputs, layerName, aSet)

 	def addOutput(self, layerName, aSet) :
 		"""Adds a mapping rule ex: .add("output1", dataset["train"])"""
		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.outputs, layerName, aSet)

 	def shuffle(self) :
 		"""Shuffles the sets. You should call this function before asking for each minibatch if you want
 		random minibacthes"""
 		if self.randomIds is None :
	 		self.randomIds = range(len(self))
 		random.shuffle(self.randomIds)

 	def _getLayerBatch(self, dct, layerName, i, size) :
 		"""This function is here because i don't like repeating myself"""
 		if not self.locked :
 			self.locked = True				

 		if size == "all" :
 			ssize = len(dct[layerName])
 		else :
 			ssize = size

 		if self.randomIds is None :
	 		return dct[layerName][i:i+ssize]
	 	else :
	 		res = []
	 		for ii in self.randomIds[i:i+ssize] :
	 			res.append(dct[layerName][ii])
	 		return res

 	def _getBatches(self, dct, i, size) :
 		"""This function is here because i don't like repeating myself"""
 		res = {}
 		for k in dct :
 			res[k] = self._getLayerBatch(dct, k, i, size)

 		return res

	def getInputLayerBatch(self, layerName, i, size) :
		"""Returns a batch for a given input layer sarting at position i, and of length size.
 		If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
 		return self._getLayerBatch(self.inputs, layerName, i, size)

 	def getInputBatches(self, i, size) :
 		"""Applies getInputLayerBatch iteratively for all layers and returns a dict:
 		layer name => batch"""
 		return self._getBatches(self.inputs, i, size)

 	def getOutputLayerBatch(self, layerName, i, size) :
		"""Returns a batch for a given output layer sarting at position i, and of length size.
 		If you want  the limit to be length of the whole set
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

class StopCriterion_ABC(object) :

	def __init__(self, *args, **kwrags) :
		self.name = self.__class__.__name__

	def check(self) :
		"""The actual function that is called by start, and the one that must be implemented in children"""
		raise NotImplemented("Must be implemented in child")

class EpochWall(StopCriterion_ABC) :
	"""Stops training when maxEpochs is reached"""
	def __init__(self, maxEpochs) :
		StopCriterion_ABC.__init__(self)
		self.maxEpochs = maxEpochs

	def check(self, trainer) :
		if trainer.currentEpoch >= self.maxEpochs :
			return True
		return False

class Trainer(object):
	"""All Trainers must inherit from me"""
	def __init__(self, cliffEpochs = 0, saveOnException = True) :
		self.cliffEpochs = cliffEpochs
		self.saveOnException = saveOnException
		
		self.bestTrainingErr = numpy.inf
		self.bestValidationErr = numpy.inf
		self.bestTestErr = numpy.inf

		self.currentEpoch = 0
		self.currentTrainingErr = numpy.inf
		self.currentValidationErr = numpy.inf
		self.currentTestErr = numpy.inf

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
			except EndOfTraining as e :
				print e.message
				death_time = time.ctime().replace(' ', '_')
				filename = "finished_" + model.name +  "_" + death_time
				model.save(filename)
			except KeyboardInterrupt, Exception :
				_dieGracefully()
				raise

	# def run(self, name, model, *args, **kwargs) :
	# 	"""The actual function that is called by start, and the one that must be implemented in children"""
	# 	raise NotImplemented("Must be implemented in child")

	# class NoEarlyStopping(Trainer) :
	# """This trainner simply never stops until you kill it. It will create a CSV file <model-name>-evolution.csv
	# to log the errors for test and train sets for each epoch, and will save the current model each time a better test 
	# error is achieved. If datasetName is provided it will be added to the evolution CSV file"""

	def run(self, name, model, trainMaps, testMaps, validationMaps, miniBatchSize, stopCriteria, shuffleMinibatches = True, datasetName = "") :
		def appendEvolution(csvEvolution, **kwargs) :			
			line = csvEvolution.newLine()
			for k, v in kwargs.iteritems() :		
				line[k] = v
			line.commit()
		
		def setHPs(layer, thing, dct) :
			try :
				thingObj = getattr(l, thing)
			except AttributeError :
				return

			if thingObj is not None :
				if type(thingObj) is types.ListType :
					for obj in thingObj :
						if len(obj.hyperParameters) == 0 :
							dct["%s_%s_%s" % (l.name, thing, obj.name)] = 1
						else :
							for hp in obj.hyperParameters :
								dct["%s_%s_%s_%s" % (l.name, thing, obj.name, hp)] = getattr(obj, hp)
				else :
					if len(thingObj.hyperParameters) == 0 :
						dct["%s_%s" % (l.name, thing)] = 1
					else :
						for hp in thingObj.hyperParameters :
							dct["%s_%s" % (l.name, hp)] = getattr(thingObj, hp)

		startTime = time.time()
		legend = ["name", "epoch", "runtime", "set", "score", "dataset_name"]
		layersForLegend = OrderedDict()

		for l in model.layers.itervalues() :
			layersForLegend["%s_size" % l.name] = len(l)
			try :
				layersForLegend["activation"] = l.activation.__name__
			except AttributeError :
				pass
			setHPs(l, "learningScenario", layersForLegend)
			setHPs(l, "costObject", layersForLegend)
			setHPs(l, "decorators", layersForLegend)

		legend.extend(layersForLegend.keys())

		csvEvolution = CSVFile(legend)
		csvEvolution.streamToFile("%s-evolution.csv" % name)

		print "learning..."
		epoch = 0
		while (nbEpochs < 0) or (epoch < nbEpochs):
			for crit in self.stopCriteria :
				if crit(self) :
					raise EndOfTraining("Training stopped because of %s" % crit.name)

			meanTrain = []
			meanTest = []
			
			for i in xrange(0, len(trainMaps), miniBatchSize) :
				if shuffleMinibatches :
					trainMaps.shuffle()
				kwargs = trainMaps.getInputBatches(i, miniBatchSize)
				for outputName in trainMaps.outputs.iterkeys() :
					kwargs.update({ "target" : trainMaps.getOutputLayerBatch(outputName, i, miniBatchSize)} )
					res = model.train(outputName, **kwargs)
					meanTrain.append(res[0])

			mt = numpy.mean(meanTrain)
			
			if len(testMaps) > 0 :	
				kwargs = testMaps.getInputBatches(0, size = "all")
				for outputName in testMaps.outputs.iterkeys() :
					kwargs.update({ "target" : testMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanTest.append(res[0])

				mte = numpy.mean(meanTest)
			else :
				mte = -1
			
			runtime = int(time.time() - startTime)
			
			if nbEpochs > 0 :
				print "epoch %s/%s (%.2f%%), mean train err %s, test err %s" %(epoch, nbEpochs, float(epoch)/nbEpochs*100, mt, mte)
			else :
				print "epoch %s/%s, mean train err %s, test err %s" %(epoch, nbEpochs, mt, mte)
			
			if mte < self.bestTestErr :
				filename = "%s-best_Testerr" % (name)
				print "\t>%s: new best test score %s -> %s" % (name, self.bestTestErr, mte)
				self.bestTestErr = mte
				if epoch > self.cliffEpochs :
					print "saving model to %s..." % (filename)
					model.save(filename)
					f = open(filename + ".txt", 'w')
					f.write("date: %s\nruntime: %s\nepoch: %s\nbest test err: %s\ntrain err: %s" % (time.ctime(), runtime, epoch, mte, mt))
					f.flush()
					f.close()

			appendEvolution(csvEvolution, 
				name = name,
				epoch = epoch,
				runtime = runtime,
				set = "Train(%d)" % len(trainMaps),
				score = mt,
				dataset_name = datasetName,
				**layersForLegend)

			appendEvolution(csvEvolution,
				name = name,
				epoch = epoch,
				runtime = runtime,
				set = "Test(%d)" % len(testMaps),
				score = mte,
				dataset_name = datasetName,
				**layersForLegend)

			sys.stdout.flush()
			epoch += 1