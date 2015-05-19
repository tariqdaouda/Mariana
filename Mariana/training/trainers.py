import cPickle, time, sys, traceback, random, types
import numpy
from collections import OrderedDict

from pyGeno.tools.parsers.CSVTools import CSVFile
import theano.tensor as tt

import Mariana.settings as MSET


class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class GGPlot2Recorder(object):
 	"""docstring for OutputRecorder"""
 	def __init__(self, verbose = True,  noBestSets = ("train", )):
		
		self.verbose = verbose
		self.noBestSets = noBestSets

 		self.outputLayers = []
		self.bestScores = {}
		self.currentScores = {}
		self.bestFilenames = {}
		self.maps = {}

		self.csvFile = None

		self._mustInit = True

	def set(self, trainer, runName, csvLegend) :
		self.trainer = trainer
		self.runName = runName
		self.csvLegend = csvLegend

	def addMap(self, mapName, mapValue) :
		if len(mapValue) > 1 :
			self.maps[mapName] = mapValue

	def addOutput(self, outputLayer) :
		self.outputLayers.append(outputLayer)

	def _init(self) :
		for o in self.outputLayers :
			self.bestScores[o.name] = {}
			self.currentScores[o.name] = {}
			self.bestFilenames[o.name] = {}
			for s, v in self.maps.iteritems() :
				self.bestScores[o.name][s] = []
				self.currentScores[o.name][s] = []
				self.bestFilenames[o.name][s] = "best_%s_%s" % (s, o.name)

		self.csvLegend.append("score")
		self.csvLegend.append("best_score")
		self.csvLegend.append("set")
		self.csvLegend.append("output")

		self.csvFile = CSVFile(legend = self.csvLegend)
		self.csvFile.streamToFile("%s-evolution.csv" % (self.runName, ), writeRate = len(self.maps) * (len(self.outputLayers) + 1) ) #(output + avg) x #set

		self._mustInit = False

	def _notify(self) :
		"""notify the recorder that there is a new entry do record"""
		self._updateScores()
		self._commitToCSVs()
		if self.verbose :
			self._printCurrentState()

	def _commitToCSVs(self) :
		"""saves the stack to disk. It will automatically add the scores and the sets to the file"""
		def _fillLine(csvFile, score, bestScore, setName, setLen, outputName, **csvValues) :
			line = csvFile.newLine()
			for k, v in csvValues.iteritems() :
				line[k] = v
			line["score"] = score
			line["best_score"] = bestScore
			line["set"] = "%s(%s)" %(setName, setLen)
			line["output"] = outputName
			line.commit()
		
		if self._mustInit :
			self._init()

		csvValues = self.store.hyperParameters

		for theSet in self.maps :
			meanCurrent = []
			meanBest = []
			for o in self.outputLayers :
				score = None
				if theSet not in self.noBestSets :
						bestScore = self.bestScores[o.name][theSet][-1]
				else :
					bestScore = self.currentScores[o.name][theSet][-1]

				score = self.currentScores[o.name][theSet][-1]
				_fillLine( self.csvFile, score, bestScore, theSet, len(self.maps[theSet]), o.name, **csvValues)

				meanCurrent.append(score)
				meanBest.append(bestScore)
		
			_fillLine( self.csvFile, numpy.mean(meanCurrent), numpy.mean(meanBest), theSet, len(self.maps[theSet]), "average", **csvValues)
		
		self.stackLen = 0

	def _updateScores(self, outputLayerName, theSet, score) :
		if self._mustInit :
			self._init()

		for outputLayerName in self.currentScores :
			for theSet in self.currentScores[outputLayerName] :
				score = self.trainer.store["scores"][outputLayerName][theSet]
				self.currentScores[outputLayerName][theSet].append(score)
			if theSet not in self.noBestSets :
				if len(self.bestScores[outputLayerName][theSet]) > 1 :
					if (score < self.bestScores[outputLayerName][theSet][-1] ) :
						self.bestScores[outputLayerName][theSet].append(score)
				else :
					self.bestScores[outputLayerName][theSet].append(score)

	def _printCurrentState(self) :
		if self.currentLen > 0 :
			print "\n=M=>State %s:" % self.currentLen
			for s in self.maps :
				print "  |-%s set" % s
				for o in self.outputLayers :
					if s not in self.noBestSets and self.currentScores[o.name][s][-1] == self.bestScores[o.name][s][-1] :
						highlight = "+best+"
					elif len(self.bestScores[o.name][s]) > 0 :
						highlight = "(best: %s)" % (self.bestScores[o.name][s][-1])
					else :
						highlight = ""

					print "    |->%s: %s %s" % (o.name, self.currentScores[o.name][s][-1], highlight)
		else :
			print "=M=> Nothing to show yet"

class DefaultTrainer(object) :

	SEQUENTIAL_TRAINING = 0
	SIMULTANEOUS_TRAINING = 1

	"""Should serve for most purposes"""
	def __init__(self, trainMaps, testMaps, validationMaps, trainMiniBatchSize, stopCriteria, testMiniBatchSize = "all", testFrequency = 1, saveOnException = True) :
		
		self.maps = {}
		self.maps["train"] = trainMaps
		self.maps["test"] = testMaps
		self.maps["validation"] = validationMaps
		
		self.miniBatchSizes = {
			"train" : trainMiniBatchSize,
			"test" : testMiniBatchSize,
			"validation" : testMiniBatchSize
		}

		self.stopCriteria = stopCriteria
		
		self.testFrequency = testFrequency
		
		self.saveOnException = saveOnException
		
		self.reset()

		self.trainingOrders = {
			self.SIMULTANEOUS_TRAINING : "SIMULTANEOUS",
			self.SEQUENTIAL_TRAINING : "SEQUENTIAL"
		}

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
		will attempt to save the model and logs. See _run documentation for a full list of possible arguments"""

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
			except KeyboardInterrupt :
				_dieGracefully()
				raise
			except :
				_dieGracefully()
				raise

	def _run(self, name, model, trainingOrder = 0, reset = True, shuffleMinibatches = True, datasetName = "") :
		"""
			trainingOrder possible values:
				* DefaultTrainer.SEQUENTIAL_TRAINING: Each output will be trained indipendetly on it's own epoch
				* DefaultTrainer.SIMULTANEOUS_TRAINING: All outputs are trained within the same epoch with the same inputs
				* Both or in O(m*n), where m is the number of mini batches and n the number of outputs
		"""

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
						dct["%s_%s" % (l.name, thingObj.name)] = 1
					else :
						for hp in thingObj.hyperParameters :
							dct["%s_%s" % (l.name, hp)] = getattr(thingObj, hp)

		def _trainTest(aMap, modelFct, shuffleMinibatches, trainingOrder, miniBatchSize) :
				if miniBatchSize == "all" :
					for outputName in aMap.getOutputNames() :
						batchData = aMap.getAll()
						kwargs = batchData[0] #inputs
						kwargs.update({ "target" : batchData[1][outputName]} )
						res = modelFct(outputName, **kwargs)
				else :
					if shuffleMinibatches :
						aMap.shuffle()
					
					vals = {}
					if trainingOrder == DefaultTrainer.SIMULTANEOUS_TRAINING :
						for i in xrange(0, len(aMap), miniBatchSize) :
							batchData = aMap.getBatches(i, miniBatchSize)
							kwargs = batchData[0] #inputs
							for outputName in aMap.getOutputNames() :
								kwargs.update({ "target" : batchData[1][outputName]} )
								res = modelFct(outputName, **kwargs)
								
								try :
									vals[outputName].append(res[0])
								except KeyError:
									vals[outputName] = [res[0]]
					
					elif trainingOrder == DefaultTrainer.SEQUENTIAL_TRAINING :
						for outputName in aMap.getOutputNames() :
							for i in xrange(0, len(aMap), miniBatchSize) :
								kwargs = batchData[0] #inputs
								batchData = aMap.getBatches(i, miniBatchSize)
								kwargs.update({ "target" : batchData[1][outputName]} )
								res = modelFct(outputName, **kwargs)

								try :
									vals[outputName].append(res[0])
								except KeyError:
									vals[outputName] = [res[0]]
					else :
						raise ValueError("Unknown training order: %s" % trainingOrder)


		if trainingOrder not in self.trainingOrders:
			raise ValueError("Unknown training order: %s" % trainingOrder)

		if reset :
			self.reset()
		
		legend = ["name", "epoch", "runtime(min)", "dataset_name", "training_order"]
		hyperParameters = OrderedDict()
		for l in model.layers.itervalues() :
			hyperParameters["%s_size" % l.name] = len(l)
			try :
				hyperParameters["activation"] = l.activation.__name__
			except AttributeError :
				pass
			setHPs(l, "learningScenario", hyperParameters)
			setHPs(l, "decorators", hyperParameters)
			if l.type == MSET.TYPE_OUTPUT_LAYER :
				setHPs(l, "costObject", hyperParameters)

		legend.extend(hyperParameters.keys())

		self.recorder = GGPlot2Recorder()
		self.recorder.set( name, legend )
		
		for m in self.maps :
			self.recorder.addMap( m, self.maps[m] )

		for l in model.outputs.itervalues() :
			self.recorder.addOutput(l)
		
		print "learning..."
		startTime = time.time()
		self.currentEpoch = 0

		while True :

		
			for mapName in ["train", "test", "validation"] :		
				aMap = self.maps[mapName]
				if len(aMap) > 0 :
					if self.miniBatchSizes[mapName] == "all" :
						for outputName in aMap.getOutputNames() :
							batchData = aMap.getAll()
							kwargs = batchData[0] #inputs
							kwargs.update({ "target" : batchData[1][outputName]} )
							if mapName == "train" :
								res = model.train(outputName, **kwargs)
							else :
								res = model.test(outputName, **kwargs)
							self.recorder.updateScore(outputName, mapName, res[0])
					else :
						if shuffleMinibatches :
							aMap.shuffle()
						vals = {}
						if trainingOrder == self.SIMULTANEOUS_TRAINING :
							for i in xrange(0, len(aMap), self.miniBatchSizes[mapName]) :
								batchData = aMap.getBatches(i, self.miniBatchSizes[mapName])
								kwargs = batchData[0] #inputs
								for outputName in aMap.getOutputNames() :
									kwargs.update({ "target" : batchData[1][outputName]} )
									if mapName == "train" :
										res = model.train(outputName, **kwargs)
									else :
										res = model.test(outputName, **kwargs)
									try :
										vals[outputName].append(res[0])
									except KeyError:
										vals[outputName] = [res[0]]
						elif trainingOrder == self.SEQUENTIAL_TRAINING :
							for outputName in aMap.getOutputNames() :
								for i in xrange(0, len(aMap), self.miniBatchSizes[mapName]) :
									kwargs = batchData[0] #inputs
									batchData = aMap.getBatches(i, self.miniBatchSizes[mapName])
									kwargs.update({ "target" : batchData[1][outputName]} )
									if mapName == "train" :
										res = model.train(outputName, **kwargs)
									else :
										res = model.test(outputName, **kwargs)
									try :
										vals[outputName].append(res[0])
									except KeyError:
										vals[outputName] = [res[0]]
						else :
							raise ValueError("Unknown training order: %s" % trainingOrder)

						for outputName in vals :
							self.recorder.updateScore(outputName, mapName, numpy.mean(vals[outputName]))

			runtime = (time.time() - startTime)/60
			
			self.store.hyperParameters = {
				"name" : name,
				"epoch" : self.currentEpoch,
				"runtime(min)" : runtime,
				"dataset_name" : datasetName,
				"training_order" : self.trainingOrders[trainingOrder]
			}

			self.store.hyperParameters.update(hyperParameters)
	
			self.recorder.commitToCSVs(**csvValues)
			self.recorder.printCurrentState()
			sys.stdout.flush()

			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise EndOfTraining(crit)

			self.currentEpoch += 1