import cPickle, time, sys, traceback, random, types
import numpy
from collections import OrderedDict

from pyGeno.tools.parsers.CSVTools import CSVFile

from Mariana.layers import *
import theano.tensor as tt

class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class DatasetMapper(object):
 	"""This class is here to map inputs of a network to datasets, or outputs to target sets.
 	If forceSameLengths all sets must have the same length."""
 	def __init__(self):
 		self.inputs = {}
 		self.outputs = {}
 		self.length = 0
 		self.randomIds = None
 		self.locked = False

 	def _add(self, dct, layer, aSet) :
 		"""This function is here because i don't like repeating myself"""
 		if self.length != 0 and len(aSet) != self.length:
 			raise ValueError("All sets must have the same number of elements. len(aSet) = %s, but another set has a length of %s" % (len(aSet), self.length))

 		if layer.name in self.inputs or layer.name in self.outputs:
 			raise ValueError("There is already a mapped layer by that name")
 		
 		if len(aSet) > self.length :
	 		self.length = len(aSet)
 		dct[layer.name] = aSet

 	def addInput(self, layer, aSet) :
 		"""Adds a mapping rule ex: .add("input1", dataset["train"])"""
 		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.inputs, layer, aSet)

 	def addOutput(self, layer, aSet) :
 		"""Adds a mapping rule ex: .add("output1", dataset["train"])"""
		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.outputs, layer, aSet)

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

	def getInputLayerBatch(self, layer, i, size) :
		"""Returns a batch for a given input layer sarting at position i, and of length size.
 		If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
 		return self._getLayerBatch(self.inputs, layer, i, size)

 	def getInputBatches(self, i, size) :
 		"""Applies getInputLayerBatch iteratively for all layers and returns a dict:
 		layer name => batch"""
 		return self._getBatches(self.inputs, i, size)

 	def getOutputLayerBatch(self, layer, i, size) :
		"""Returns a batch for a given output layer sarting at position i, and of length size.
 		If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
 		return self._getLayerBatch(self.outputs, layer, i, size)
 	
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

	def stop(self) :
		"""The actual function that is called by start, and the one that must be implemented in children"""
		raise NotImplemented("Must be implemented in child")

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return self.name

class EpochWall(StopCriterion_ABC) :
	"""Stops training when maxEpochs is reached"""
	def __init__(self, maxEpochs) :
		StopCriterion_ABC.__init__(self)
		self.maxEpochs = maxEpochs

	def stop(self, trainer) :
		if trainer.currentEpoch >= self.maxEpochs :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached epoch wall %s" % self.maxEpochs

class TestScoreWall(StopCriterion_ABC) :
	"""Stops training when maxEpochs is reached"""
	def __init__(self, wallValue) :
		StopCriterion_ABC.__init__(self)
		self.wallValue = wallValue

	def stop(self, trainer) :
		if trainer.currentTestScore <= self.wallValue :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached test score wall %s" % self.wallValue

class GeometricEarlyStopping(StopCriterion_ABC) :
	"""Stops training when maxEpochs is reached"""
	def __init__(self, patience, significantImprovement) :
		StopCriterion_ABC.__init__(self)
		self.patienceIncrease = patience
		self.wall = patience
		self.significantImprovement = significantImprovement

	def stop(self, trainer) :
		if self.wall <= 0 :
			return True

		if trainer.currentValidationScore < (trainer.bestValidationScore * self.significantImprovement) :
			self.wall += (trainer.currentEpoch * self.patienceIncrease)
		self.wall -= 1	
		return False
	
	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Early stopping, no patience left"

class Trainer(object):
	"""The trainer"""
	def __init__(self, trainMaps, testMaps, validationMaps, miniBatchSize, stopCriteria, testFrequency = 1, validationFrequency = 1, saveOnException = True) :
		
		self.trainMaps = trainMaps
		self.testMaps = testMaps
		self.validationMaps = validationMaps

		self.miniBatchSize = miniBatchSize
		self.stopCriteria = stopCriteria
		
		self.testFrequency = testFrequency
		self.validationFrequency = validationFrequency

		self.saveOnException = saveOnException
		
		self.reset()

	def reset(self) :
		'resets the bests'
		
		self.bestTrainingScore = numpy.inf
		self.bestValidationScore = numpy.inf
		self.bestTestScore = numpy.inf
		
		self.bestTestModelFile = None
		self.bestValidationModelFile = None

	def getBestValidationModel(self) :
		"""loads the best validation model from disk and returns it"""
		f = open(self.bestValidationModelFile + ".mariana.pkl")
		model = cPickle.load(f) 
		f.close()
		return model

	def getBestTestModel(self) :
		"""loads the best Test model from disk and returns it"""
		f = open(self.bestTestModelFile + ".mariana.pkl")
		model = cPickle.load(f)
		f.close()
		return model
		
	def start(self, name, model, *args, **kwargs) :
		"""Starts the training. If anything bad and unexpcted happens during training, the Trainer
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
			f.flush()
			f.close()

		self.currentEpoch = 0
		self.currentTrainingScore = numpy.inf
		self.currentValidationScore = numpy.inf
		self.currentTestScore = numpy.inf

		if not self.saveOnException :
			return self._run(name, model, *args, **kwargs)
		else :
			try :
				return self._run(name, model, *args, **kwargs)
			except EndOfTraining as e :
				print e.message
				death_time = time.ctime().replace(' ', '_')
				filename = "finished_" + name +  "_" + death_time
				f = open(filename +  ".stopreason.txt", 'w')
				f.write("Time of death: %s\n" % death_time)
				f.write("Stopped by: %s\n" % e.stopCriterion.name)
				f.write("Reason: %s\n" % e.message)
				f.flush()
				f.close()
				model.save(filename)
			except KeyboardInterrupt, Exception :
				_dieGracefully()
				raise

	def _run(self, name, model, reset = True, shuffleMinibatches = True, datasetName = "") :
				
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

		if reset :
			self.reset()
		
		startTime = time.time()
		legend = ["name", "epoch", "runtime", "set", "score", "dataset_name"]
		layersForLegend = OrderedDict()
		outputScores = {}

		for l in model.layers.itervalues() :
			if l.type == "output" :
				for typ in ["train", "test", "validation"] :
					scoreName = "%s_%s_score" % (l.name, typ)
					legend.append( scoreName )
					outputScores[ scoreName ] = -1
			
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
		csvEvolution.streamToFile("%s-evolution.csv" % name, writeRate = 1)

		csvBestTestScore = CSVFile(legend)
		csvBestTestScore.streamToFile("%s-best_Test_Score.csv" % (name), writeRate = 1)
		csvBestValidationScore = CSVFile(legend)
		csvBestValidationScore.streamToFile("%s-best_Validation_Score.csv" % (name), writeRate = 1)

		print "learning..."
		self.currentEpoch = 0
		while True :
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise EndOfTraining(crit)

			meanTrain = []
			meanTest = []
			meanValidation = []
			
			for i in xrange(0, len(self.trainMaps), self.miniBatchSize) :
				if shuffleMinibatches :
					self.trainMaps.shuffle()
				kwargs = self.trainMaps.getInputBatches(i, self.miniBatchSize)
				for outputName in self.trainMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.trainMaps.getOutputLayerBatch(outputName, i, self.miniBatchSize)} )
					res = model.train(outputName, **kwargs)
					meanTrain.append(res[0])
					outputScores["%s_%s_score" % (outputName, "train")] = res[0]

			self.currentTrainingScore = numpy.mean(meanTrain)
			
			if len(self.testMaps) > 0 and (self.currentEpoch % self.testFrequency == 0) :
				kwargs = self.testMaps.getInputBatches(0, size = "all")
				for outputName in self.testMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.testMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanTest.append(res[0])
					outputScores["%s_%s_score" % (outputName, "test")] = res[0]

				self.currentTestScore = numpy.mean(meanTest)
			else :
				self.currentTestScore = -1
			
			if len(self.validationMaps) > 0 and (self.currentEpoch % self.validationFrequency == 0) :
				kwargs = self.validationMaps.getInputBatches(0, size = "all")
				for outputName in self.validationMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.validationMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanValidation.append(res[0])
					outputScores["%s_%s_score" % (outputName, "validation")] = res[0]

				self.currentValidationScore = numpy.mean(meanValidation)
			else :
				self.currentValidationScore = -1

			runtime = (time.time() - startTime)/60
			
			print "epoch %s, mean Score (train: %s, test: %s, validation: %s)" %(self.currentEpoch, self.currentTrainingScore, self.currentTestScore, self.currentValidationScore)
			
			if self.currentTestScore < self.bestTestScore :
				print "\t===>%s: new best test score %s -> %s" % (name, self.bestTestScore, self.currentTestScore)
				self.bestTestScore = self.currentTestScore
				params = {
					"name" : name,
					"epoch" : self.currentEpoch,
					"runtime" : runtime,
					"set" : "Test(%d)" % len(self.testMaps),
					"score" : self.currentTestScore,
					"dataset_name" : datasetName,
				}
				params.update(layersForLegend)
				params.update(outputScores)
				appendEvolution(csvBestTestScore, **params)

			if self.currentValidationScore < self.bestValidationScore :
				print "\txxx>%s: new best Validation score %s -> %s" % (name, self.bestValidationScore, self.currentValidationScore)
				self.bestValidationScore = self.currentValidationScore
				params = {
					"name" : name,
					"epoch" : self.currentEpoch,
					"runtime" : runtime,
					"set" : "Test(%d)" % len(self.validationMaps),
					"score" : self.currentValidationScore,
					"dataset_name" : datasetName,
				}
				params.update(layersForLegend)
				params.update(outputScores)
				appendEvolution(csvBestValidationScore, **params)

			params = {
					"name" : name,
					"epoch" : self.currentEpoch,
					"runtime" : runtime,
					"set" : "Train(%d)" % len(self.trainMaps),
					"score" : self.currentTrainingScore,
					"dataset_name" : datasetName,
				}
			params.update(layersForLegend)
			params.update(outputScores)
			appendEvolution(csvEvolution, **params)

			params["set"] = "Test(%d)" % len(self.testMaps)
			params["score"] = self.currentTrainingScore
			appendEvolution(csvEvolution, **params)

			params["set"] = "Validation(%d)" % len(self.validationMaps)
			params["score"] = self.currentValidationScore
			appendEvolution(csvEvolution, **params)

			sys.stdout.flush()
			self.currentEpoch += 1