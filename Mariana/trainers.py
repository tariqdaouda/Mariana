import cPickle, time, sys, traceback, random, types
import numpy
from collections import OrderedDict

from pyGeno.tools.parsers.CSVTools import CSVFile

import Mariana.layers as ML
import theano.tensor as tt

class DatasetMapper(object):
 	"""This class is here to map inputs of a network to datasets, or outputs to target sets.
 	If forceSameLengths all sets must have the same length."""
 	def __init__(self):
 		self.inputs = {}
 		self.outputs = {}
 		self.length = 0
 		self.randomIds = None
 		self.locked = False

 	def _add(self, dct, layer, setOrLayer) :
 		"""This function is here because i don't like repeating myself"""
 		if self.length != 0 and len(setOrLayer) != self.length:
 			raise ValueError("All sets must have the same number of elements. len(setOrLayer) = %s, but another set has a length of %s" % (len(setOrLayer), self.length))

 		if layer.name in self.inputs or layer.name in self.outputs:
 			raise ValueError("There is already a mapped layer by that name")
 		
 		if len(setOrLayer) > self.length :
	 		self.length = len(setOrLayer)
 		dct[layer.name] = setOrLayer

 	def addInput(self, layer, setOrLayer) :
 		"""Adds a mapping rule ex: .add(input1, dataset["train"]["examples"])"""
 		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.inputs, layer, setOrLayer)

 	def addOutput(self, layer, setOrLayer) :
 		"""Adds a mapping rule ex: .add(output1, dataset["train"]["targets"])"""
		if self.locked :
 			raise ValueError("Can't add a map if a batch has already been requested")
 		self._add(self.outputs, layer, setOrLayer)

 	def shuffle(self) :
 		"""Shuffles the sets. You should call this function before asking for each minibatch if you want
 		random minibatches"""
 		if self.randomIds is None :
	 		self.randomIds = range(len(self))
 		random.shuffle(self.randomIds)

 	def _getLayerBatch(self, dct, layerName, i, size) :
 		"""This function is here because i don't like repeating myself"""
 		if not self.locked :
 			self.locked = True				

 		if issubclass(dct[layerName].__class__, ML.Layer_ABC) :
 			return dct[layerName].outputs

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

class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class TrainingRecorder(object):
 	"""docstring for OutputRecorder"""
 	def __init__(self, runName, csvLegend):
		self.runName = runName
		self.csvLegend = csvLegend

 		self.outputLayers = []
		self.bestScores = {}
		self.currentScores = {}
		self.bestFilenames = {}
		self.maps = {}

		self.stackLen = 0
		self.currentLen = 0
		self.stackLen = 0
		self.csvFile = None

		self._mustInit = True
	def addMap(self, mapName, mapValue) :
		self.maps[mapName] = mapValue

	def addOutput(self, outputLayer) :
		self.outputLayers.append(outputLayer)

	def _init(self) :
		for o in self.outputLayers :
			for s in self.maps :
				self.bestScores[o][s] = []
				self.currentScores[o][s] = []
				self.bestFilenames[o][s] = "best_%s_%s" % (s, o.name)

		self.csvLegend.append("score")
		self.csvLegend.append("best_score")
		self.csvLegend.append("set")
		self.csvLegend.append("output")
		self.csvFile = CSVFile(legend = self.csvLegend)
		self.csvFile.streamToFile("%s-evolution.csv" % (runName, ), writeRate = len(self.maps) * (len(self.outputLayers) + 1) ) #(output + avg) x #set

		self._mustInit = False

	def getBestScore(self, outputLayer, theSet) :
		return self.bestScores[outputLayers.name][theSet][-1]

	def getCurrentScore(self, outputLayer, theSet) :
		return self.currentScores[outputLayers.name][theSet][-1]

	def commitToCSVs(self, **csvValues) :
		"""saves the stack to disk. It will automatically add the scores and the sets to the file"""
		def _fillLine(line, score, bestScore, setName, setLen, outputName, **csvValues) :
			for k, v in csvValues.iteritems() :
				line[k] = v
			line["score"] = score
			line["best_score"] = bestScore
			line["set"] = "%s(%s)" %(setName, setName)
			line["output"] = outputName
			line.commit()
		
		if self._mustInit :
			self._init()

		start = self.currentLen - self.stackLen
		stop = self.currentLen
		for i in xrange(start, stop) :
			meanCurrent = []
			meanBest = []
			for theSet in self.maps :
				for o in self.outputLayers :
					score = None
					try :
						bestScore = self.bestScores[o.name][theSet][i]
					except IndexError :
						bestScore = self.bestScores[o.name][theSet][-1]
					
					score = self.currentScores[o.name][theSet][i]
					_fillLine( self.csvFile.newLine(), score, bestScore, theSet, len(self.maps[theSet]), o.name, **csvValues)

					meanCurrent.append(self.currentScores[o.name][theSet][i])	
					meanBest.append(self.bestScores[o.name][theSet][i])	
			
				meanCurrent = numpy.mean(meanCurrent)
				meanBest = numpy.mean(meanBest)

				_fillLine( self.csvFile.newLine(), meanCurrent, meanBest, theSet, len(self.maps[theSet]), "average", **csvValues)
		
		self.stackLen = 0

	def updateScore(self, outputLayerName, theSet, score) :
		self.currentScores[outputLayerName][theSet].append(score)
		if score < self.bestScores[outputLayerName][theSet] and theSet != "train":
			print "\t=x=> new best [-%s-] score for layer '%s' %s -> %s [:-)" % (theSet.upper(), self.outputLayerName, self.bestScores[outputLayerName][theSet], score)
			self.bestScores[outputLayerName][theSet].append(score)
		self.stackLen += 1
		self.currentLen += 1 

class DefaultTrainer(object):
	"""Should serve for most purposes"""
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
		'resets the beast'	
		
		self.recorder = None
		self.currentEpoch = 0

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
		"""Starts the training. If anything bad and unexpected happens during training, the Trainer
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
				f.write("Epoch of death: %s\n" % self.currentEpoch)
				f.write("Stopped by: %s\n" % e.stopCriterion.name)
				f.write("Reason: %s\n" % e.message)
				f.flush()
				f.close()
				model.save(filename)
			except KeyboardInterrupt, Exception :
				_dieGracefully()
				raise

	def _run(self, name, model, reset = True, shuffleMinibatches = True, datasetName = "") :
		
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
		
		self.recorder = TrainingRecorder( name, legend )
		
		for m, v in [ ( "train", self.trainMaps ), ( "test", self.testMaps ), ( "validation", self.validationMaps )] :
			self.recorder.addMap( m, v )

		legend = ["name", "epoch", "runtime", "set", "score", "dataset_name"]
		layersForLegend = OrderedDict()
		for l in model.outputs.itervalues() :			
			self.recorder.addOutput(l)	
			layersForLegend["%s_size" % l.name] = len(l)
			try :
				layersForLegend["activation"] = l.activation.__name__
			except AttributeError :
				pass
			setHPs(l, "learningScenario", layersForLegend)
			setHPs(l, "costObject", layersForLegend)
			setHPs(l, "decorators", layersForLegend)

		legend.extend(layersForLegend.keys())


		print "learning..."
		startTime = time.time()
		self.currentEpoch = 0
		while True :
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise EndOfTraining(crit)

			for k in outputScores :
				outputScores[k] = []

			for i in xrange(0, len(self.trainMaps), self.miniBatchSize) :
				if shuffleMinibatches :
					self.trainMaps.shuffle()

				kwargs = self.trainMaps.getInputBatches(i, self.miniBatchSize)
				for outputName in self.trainMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.trainMaps.getOutputLayerBatch(outputName, i, self.miniBatchSize)} )
					res = model.train(outputName, **kwargs)
					meanTrain.append(res[0])
					self.recorder.updateScore(outputName, "train", res[0])
			
			if len(self.testMaps) > 0 and (self.currentEpoch % self.testFrequency == 0) :
				kwargs = self.testMaps.getInputBatches(0, size = "all")
				for outputName in self.testMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.testMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanTest.append(res[0])
					self.recorder.updateScore(outputName, "test", res[0])
			
			if len(self.validationMaps) > 0 and (self.currentEpoch % self.validationFrequency == 0) :
				kwargs = self.validationMaps.getInputBatches(0, size = "all")
				for outputName in self.validationMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.validationMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanValidation.append(res[0])
					self.recorder.updateScore(outputName, "validation", res[0])

			runtime = (time.time() - startTime)/60
			
			csvValues = {
				"name" : name,
				"epoch" : self.currentEpoch,
				"runtime(min)" : runtime,
				"dataset_name" : datasetName,
			}
			csvValues.update(layersForLegend)
			csvValues.update(outputScores)

			self.recorder.commitToCSVs(**csvValues)

			sys.stdout.flush()
			self.currentEpoch += 1