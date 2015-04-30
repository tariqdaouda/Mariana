import cPickle, time, sys, traceback, random, types
import numpy
from collections import OrderedDict

from pyGeno.tools.parsers.CSVTools import CSVFile

import Mariana.layers as ML
import theano.tensor as tt

class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class TrainingRecorder(object):
 	"""docstring for TrainingRecorder"""
 	def __init__(self, runName, outputLayerName, csvLegend):
 		super(TrainingRecorder, self).__init__()
 		self.outputLayerName = outputLayerName
		self.runName = runName

		self.csvFiles = {
			"test" : CSVFile(legend = csvLegend),
			"train" : CSVFile(legend = csvLegend),
			"validation" : CSVFile(legend = csvLegend),
		}

		for k in self.csvFiles :
			self.csvFiles.streamToFile("%s-%s-best_%s_score.csv" % (runName, outputLayerName, k) , writeRate = 1)
			
		self.csvEvolution = CSVFile(legend = csvLegend)
		self.csvEvolution.streamToFile("%s-%s-evolution.csv" % (runName, outputLayerName) , writeRate = 1)

		self.bestScores = {
			"test" : None,
			"train" : None,
			"validation" : None
		}
		
		self.bestModelFiles = {
			"test" : None,
			"train" : None,
			"validation" : None
		}
		
		self.currentScores = {
			"test" : numpy.inf,
			"train" : numpy.inf,
			"validation" : numpy.inf	
		}

	def _appendEvolution(self, csvEvolution, **kwargs) :
		line = csvEvolution.newLine()
		for k, v in kwargs.iteritems() :
			line[k] = v
		line.commit()

	def updateEvolution(self, **csvValues) :
		self._appendEvolution(self.csvEvolution, **csvValues)

	def updateBestScore(self, theSet, score, **csvValues) :
		self.currentScores[theSet] = score
		if score < self.bestScores[theSet] :
			print "\t=x=> new best [-%s-] score for layer '%s' %s -> %s [:-)" % (theSet.upper(), self.outputLayerName, self.bestScores[theSet], score)
			self.bestScores[theSet] = score
			self._appendEvolution(self.bestScores[theSet], **csvValues)


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
	
		self.bestTrainingScores = {}
		self.bestValidationScores = {}
		self.bestTestScores = {}

		self.bestTestModelFiles = {}
		self.bestValidationModelFiles = {}
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
		# self.currentTrainingScore = numpy.inf
		# self.currentValidationScore = numpy.inf
		# self.currentTestScore = numpy.inf

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
		
		startTime = time.time()
		legend = ["name", "epoch", "runtime", "set", "score", "dataset_name"]
		layersForLegend = OrderedDict()
		
		for l in model.outputs.itervalues() :
			if l.type == "output" :
				for typ in ["train", "test", "validation"] :
					scoreName = "%s_%s_score" % (l.name, typ)
					legend.append( scoreName )
					outputScores[ scoreName ] = []
			
			layersForLegend["%s_size" % l.name] = len(l)
			try :
				layersForLegend["activation"] = l.activation.__name__
			except AttributeError :
				pass
			setHPs(l, "learningScenario", layersForLegend)
			setHPs(l, "costObject", layersForLegend)
			setHPs(l, "decorators", layersForLegend)

		legend.extend(layersForLegend.keys())

		self.recorders = {}
		for l in model.outputs.itervalues() :
			self.recorders[l.name] = TrainingRecorder( name, l.name, legend )

		self.recorders["mean"] = TrainingRecorder( name, "mean", legend )

		print "learning..."
		self.currentEpoch = 0
		while True :
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise EndOfTraining(crit)

			csvFields = {
					"name" : name,
					"epoch" : self.currentEpoch,
					"runtime" : runtime,
					"set" : None,
					"score" : None,
					"dataset_name" : datasetName,
				}
			csvFields.update(layersForLegend)
			csvFields.update(outputScores)

			meanTrain = []
			meanTest = []
			meanValidation = []
			
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
					outputScores["%s_%s_score" % (outputName, "train")].append(res[0])

			self.currentTrainingScore = numpy.mean(meanTrain)
			
			if len(self.testMaps) > 0 and (self.currentEpoch % self.testFrequency == 0) :
				kwargs = self.testMaps.getInputBatches(0, size = "all")
				for outputName in self.testMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.testMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanTest.append(res[0])
					outputScores["%s_%s_score" % (outputName, "test")].append(res[0])

				self.currentTestScore = numpy.mean(meanTest)
			else :
				self.currentTestScore = 0
			
			if len(self.validationMaps) > 0 and (self.currentEpoch % self.validationFrequency == 0) :
				kwargs = self.validationMaps.getInputBatches(0, size = "all")
				for outputName in self.validationMaps.outputs.iterkeys() :
					kwargs.update({ "target" : self.validationMaps.getOutputLayerBatch(outputName, 0, size = "all")} )
					res = model.test(outputName, **kwargs)
					meanValidation.append(res[0])
					outputScores["%s_%s_score" % (outputName, "validation")].append(res[0])

				self.currentValidationScore = numpy.mean(meanValidation)
			else :
				self.currentValidationScore = 0

			runtime = (time.time() - startTime)/60
			
			for k in outputScores :
				outputScores[k] = numpy.mean(outputScores[k])

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
			params["score"] = self.currentTestScore
			appendEvolution(csvEvolution, **params)

			params["set"] = "Validation(%d)" % len(self.validationMaps)
			params["score"] = self.currentValidationScore
			appendEvolution(csvEvolution, **params)

			sys.stdout.flush()
			self.currentEpoch += 1