import cPickle, time, sys, os, traceback, types, json, signal
import numpy
from collections import OrderedDict
import theano.tensor as tt

import Mariana.settings as MSET
import Mariana.training.recoders as MREC
import Mariana.candies as MCAN


class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class DefaultTrainer(object) :

	SEQUENTIAL_TRAINING = 0
	SIMULTANEOUS_TRAINING = 1

	"""Should serve for most purposes"""
	def __init__(self,
		trainMaps,
		testMaps,
		validationMaps,
		trainMiniBatchSize,
		stopCriteria,
		testMiniBatchSize = "all",
		validationMiniBatchSize = "all",
		testFrequency = 1,
		saveOnException = True) :
		
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
		
		self.store = {}
		self.store["runInfos"] = {}
		self.store["scores"] = {}
		self.store["hyperParameters"] = {}
		self.store["setSizes"] = {}

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
		
	def start(self, runName, model, recorder = "default", **kwargs) :
		"""Starts the training. If anything bad and unexpected happens during training, the Trainer
		will attempt to save the model and logs. See _run documentation for a full list of possible arguments"""
		import json, signal

		def _handler_sig_term(sig, frame) :
			_dieGracefully("SIGTERM", None)
			sys.exit(sig)

		def _dieGracefully(exType, tb = None) :
			# traceback.print_tb(tb)
			if type(exType) is types.StringType :
				exName = exType
			else :
				exName = exType.__name__

			death_time = time.ctime().replace(' ', '_')
			filename = "dx-xb_" + runName + "_death_by_" + exName + "_" + death_time
			sys.stderr.write("\n===\nDying gracefully from %s, and saving myself to:\n...%s\n===\n" % (exName, filename))
			model.save(filename)
			f = open(filename +  ".traceback.log", 'w')
			f.write("Mariana training Interruption\n=============================\n")
			f.write("\nDetails\n-------\n")
			f.write("Name: %s\n" % runName)
			f.write("pid: %s\n" % os.getpid())
			f.write("Killed by: %s\n" % str(exType))
			f.write("Time of death: %s\n" % death_time)
			f.write("Model saved to: %s\n" % filename)
			
			if tb is not None :
				f.write("\nTraceback\n---------\n")
				f.write(str(traceback.extract_tb(tb)).replace("), (", "),\n(").replace("[(","[\n(").replace(")]",")\n]"))
			f.flush()
			f.close()

		signal.signal(signal.SIGTERM, _handler_sig_term)
		print "\n" + "Training starts."
		MCAN.friendly("Process id", "The pid of this run is: %d" % os.getpid())

		if recorder == "default" :
			recorder = MREC.GGPlot2(runName, verbose = True)
			MCAN.friendly(
				"Default recorder",
				"The trainer will recruit the default 'GGPlot2' recorder on verbose mode.\nResults will be saved into '%s'." % (recorder.filename)
				)
		
		self.currentEpoch = 0

		if not self.saveOnException :
			return self._run(runName, model, recorder, **kwargs)
		else :
			try :
				return self._run(runName, model, recorder, **kwargs)
			except EndOfTraining as e :
				print e.message
				death_time = time.ctime().replace(' ', '_')
				filename = "finished_" + runName +  "_" + death_time
				f = open(filename +  ".stopreason.txt", 'w')
				f.write("Name: %s\n" % runName)
				f.write("pid: %s\n" % os.getpid())
				f.write("Time of death: %s\n" % death_time)
				f.write("Epoch of death: %s\n" % self.currentEpoch)
				f.write("Stopped by: %s\n" % e.stopCriterion.name)
				f.write("Reason: %s\n" % e.message)
				sstore = str(self.store).replace("'", '"')
				f.write(
					json.dumps(
						json.loads(sstore), sort_keys=True,
						indent=4,
						separators=(',', ': ')
					)
				)

				f.flush()
				f.close()
				model.save(filename)
			except KeyboardInterrupt :
				exType, ex, tb = sys.exc_info()
				_dieGracefully(exType, tb)
				raise
			except :
				exType, ex, tb = sys.exc_info()
				_dieGracefully(exType, tb)
				raise

	def _run(self, name, model, recorder, trainingOrder = 0, reset = True, shuffleMinibatches = False, datasetName = "") :
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
			scores = {}
			if miniBatchSize == "all" :
				for output in aMap.outputLayers :
					kwargs = dict( aMap.getAll() )
					kwargs["target"] = kwargs[output.name]
					del(kwargs[output.name])
					res = modelFct(output, **kwargs)
					scores[output.name] = res[0]
			else :
				
				if trainingOrder == DefaultTrainer.SEQUENTIAL_TRAINING :
					for output in aMap.outputLayers :
						for i in xrange(0, len(aMap), miniBatchSize) :
							batchData = aMap.getBatch(i, miniBatchSize)
							batchData["target"] = batchData[output.name]
							del(batchData[output.name])
							res = modelFct(output, **batchData)
							try :
								scores[output.name].append(res[0])
							except KeyError:
								scores[output.name] = [res[0]]
				elif trainingOrder == DefaultTrainer.SIMULTANEOUS_TRAINING :
					for i in xrange(0, len(aMap), miniBatchSize) :
						batchData = aMap.getBatch(i, miniBatchSize)
						for output in aMap.outputLayers :
							batchData["target"] = batchData[output.name]
							del(batchData[output.name])
							res = modelFct(output, **batchData)
							
							try :
								scores[output.name].append(res[0])
							except KeyError:
								scores[output.name] = [res[0]]		
				else :
					raise ValueError("Unknown training order: %s" % trainingOrder)

			if len(scores) > 1 :
				scores["average"] = 0
				for outputName in scores :
					scores[outputName] = numpy.mean(scores[outputName])
					scores["average"] += scores[outputName]

				scores["average"] = numpy.mean(scores["average"])
			else :
				for outputName in scores :
					scores[outputName] = numpy.mean(scores[outputName])

			return scores

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
		self.store["hyperParameters"].update(hyperParameters)
		
		self.store["runInfos"]["name"] = name
		self.store["runInfos"]["dataset_name"] = datasetName
		self.store["runInfos"]["training_order"] = self.trainingOrders[trainingOrder]
		self.store["runInfos"]["pid"] = os.getpid()

		for mapName in self.maps :
			self.store["setSizes"][mapName] = len(self.maps[mapName])

		self.recorder = recorder
		
		startTime = time.time()
		self.currentEpoch = 0

		while True :
			for mapName in ["train", "test", "validation"] :
				aMap = self.maps[mapName]
				if len(aMap) > 0 :			
					if shuffleMinibatches :
						aMap.reroll()
					if mapName == "train" :
						modelFct = model.train
					else :
						modelFct = model.test
					
					self.store["scores"][mapName] = _trainTest(
						aMap,
						modelFct,
						shuffleMinibatches,
						trainingOrder,
						self.miniBatchSizes[mapName]
					)

			runtime = (time.time() - startTime)/60
			
			self.store["runInfos"].update( (
				("epoch", self.currentEpoch),
				("runtime(min)", runtime),
			) )
	
			self.recorder.commit(self.store, model)
			
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise EndOfTraining(crit)

			self.currentEpoch += 1