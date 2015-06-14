import cPickle, time, sys, os, traceback, types, json, signal
import numpy
from collections import OrderedDict
import theano.tensor as tt

import Mariana.layers as ML
import Mariana.training.recorders as MREC
import Mariana.training.stopcriteria as MSTOP
import Mariana.candies as MCAN

__all__ = ["Trainer_ABC", "DefaultTrainer"]

class Trainer_ABC(object) :
	"""This is the general interface of trainer"""

	def __init__(self) :
		"""
		The store is initialised to::

			self.store = {
				"runInfos" : {
					"epoch" : 0,
				},
				"scores" : {},
				"hyperParameters" : {},
				"setSizes" : {},
			}

		"""
		self.store = {
			"runInfos" : {
				"epoch" : 0,
			},
			"scores" : {},
			"hyperParameters" : {},
			"setSizes" : {},
		}

	def start(self, runName, model, recorder, *args, **kwargs) :
		"""Starts the training and encapsulates it into a safe environement.
		If the training stops because of an Exception or SIGTEM, the trainer
		will save logs, the store, and the last version of the model.
		"""

		import json, signal, cPickle

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
			sstore = str(self.store).replace("'", '"')
			f.write(
				"store:\n%s" % json.dumps(
					json.loads(sstore), sort_keys=True,
					indent=4,
					separators=(',', ': ')
				)
			)

			if tb is not None :
				f.write("\nTraceback\n---------\n")
				f.write(str(traceback.extract_tb(tb)).replace("), (", "),\n(").replace("[(","[\n(").replace(")]",")\n]"))
			f.flush()
			f.close()
			f = open(filename + ".store.pkl", "wb")
			cPickle.dump(self.store, f)
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
		
		if not self.saveIfMurdered :
			return self.run(runName, model, recorder, *args, **kwargs)
		else :
			try :
				return self.run(runName, model, recorder, *args, **kwargs)
			except MSTOP.EndOfTraining as e :
				print e.message
				death_time = time.ctime().replace(' ', '_')
				filename = "finished_" + runName +  "_" + death_time
				f = open(filename +  ".stopreason.txt", 'w')
				f.write("Name: %s\n" % runName)
				f.write("pid: %s\n" % os.getpid())
				f.write("Time of death: %s\n" % death_time)
				f.write("Epoch of death: %s\n" % self.store["runInfos"]["epoch"])
				f.write("Stopped by: %s\n" % e.stopCriterion.name)
				f.write("Reason: %s\n" % e.message)
				sstore = str(self.store).replace("'", '"')
				f.write(
					"store:\n%s" % json.dumps(
						json.loads(sstore), sort_keys=True,
						indent=4,
						separators=(',', ': ')
					)
				)

				f.flush()
				f.close()
				model.save(filename)
				f = open(filename + ".store.pkl", "wb")
				cPickle.dump(self.store, f)
				f.close()

			except KeyboardInterrupt :
				exType, ex, tb = sys.exc_info()
				_dieGracefully(exType, tb)
				raise
			except :
				exType, ex, tb = sys.exc_info()
				_dieGracefully(exType, tb)
				raise

	def run(self, *args, **kwargs) :
		"""Abtract function must be implemented in child. This function should implement the whole training process"""
		raise NotImplemented("Must be implemented in child")

class DefaultTrainer(Trainer_ABC) :
	"""The default trainer should serve for most purposes"""

	SEQUENTIAL_TRAINING = 0
	SIMULTANEOUS_TRAINING = 1
	ALL_SET = -1

	def __init__(self,
		trainMaps,
		testMaps,
		validationMaps,
		trainMiniBatchSize,
		stopCriteria,
		testMiniBatchSize = -1,
		validationMiniBatchSize = -1,
		saveIfMurdered = True) :
		"""
			:param DatasetMaps trainMaps: Layer mappings for the training set
			:param DatasetMaps testtrainMaps: Layer mappings for the testing set
			:param DatasetMaps validationMaps: Layer mappings for the validation set, use DefaultTrainer.ALL_SET for the whole set
			:param int trainMiniBatchSize: The size of a training minibatch, use DefaultTrainer.ALL_SET for the whole set
			:param list stopCriteria: List of StopCriterion objects 
			:param int testMiniBatchSize: The size of a testing minibatch, use DefaultTrainer.ALL_SET for the whole set
			:param int validationMiniBatchSize: The size of a validationMiniBatchSize minibatch
			:param bool saveIfMurdered: Die gracefully in case of Exception or SIGTERM and save the current state of the model and logs
		"""
		
		Trainer_ABC.__init__(self)

		self.maps = {
			"train": trainMaps,
			"test": testMaps,
			"validation": validationMaps
		}

		self.miniBatchSizes = {
			"train" : trainMiniBatchSize,
			"test" : testMiniBatchSize,
			"validation" : testMiniBatchSize
		}

		self.stopCriteria = stopCriteria		
		self.saveIfMurdered = saveIfMurdered
		
		self.trainingOrdersHR = {
			self.SIMULTANEOUS_TRAINING : "SIMULTANEOUS",
			self.SEQUENTIAL_TRAINING : "SEQUENTIAL"
		}

	def start(self, runName, model, recorder = "default", trainingOrder = 0, shuffle = False, datasetName = "") :
		"""starts the training, cf. run() for the a description of the arguments"""
		Trainer_ABC.start( self, runName, model, recorder, trainingOrder, shuffle, datasetName )

	def run(self, name, model, recorder, trainingOrder, shuffle, datasetName) :
		"""
			:param str runName: The name of this run
			:param Recorder recorder: A recorder object
			:param int trainingOrder:
				* DefaultTrainer.SEQUENTIAL_TRAINING: Each output will be trained indipendetly on it's own epoch
				* DefaultTrainer.SIMULTANEOUS_TRAINING: All outputs are trained within the same epoch with the same inputs
				* Both are in O(m*n), where m is the number of mini batches and n the number of outputs
			:param bool reset: Should the trainer be reset before starting the run
			:param bool shuffle: Should the datasets be shuffled at each epoch
			:param str datasetName: If provided, the name of the dataset will be stored as a hyper parameter
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

		def _trainTest(aMap, modelFct, shuffle, trainingOrder, miniBatchSize) :
			scores = {}
			if miniBatchSize == DefaultTrainer.ALL_SET :
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

		if trainingOrder not in self.trainingOrdersHR:
			raise ValueError("Unknown training order: %s" % trainingOrder)
		
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
			if l.type == ML.TYPE_OUTPUT_LAYER :
				setHPs(l, "costObject", hyperParameters)

		legend.extend(hyperParameters.keys())
		self.store["hyperParameters"].update(hyperParameters)
		
		self.store["runInfos"]["name"] = name
		self.store["runInfos"]["dataset_name"] = datasetName
		self.store["runInfos"]["training_order"] = self.trainingOrdersHR[trainingOrder]
		self.store["runInfos"]["pid"] = os.getpid()

		for mapName in self.maps :
			self.store["setSizes"][mapName] = len(self.maps[mapName])

		self.recorder = recorder
		
		startTime = time.time()
		self.store["runInfos"]["epoch"] = 0
		while True :
			for mapName in ["train", "test", "validation"] :
				aMap = self.maps[mapName]
				if len(aMap) > 0 :			
					if shuffle :
						aMap.reroll()
					if mapName == "train" :
						modelFct = model.train
					else :
						modelFct = model.test
					
					self.store["scores"][mapName] = _trainTest(
						aMap,
						modelFct,
						shuffle,
						trainingOrder,
						self.miniBatchSizes[mapName]
					)
			
			runtime = (time.time() - startTime)/60
			
			self.store["runInfos"].update( (
				("runtime(min)", runtime),
			) )
	
			self.recorder.commit(self.store, model)
			
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise MSTOP.EndOfTraining(crit)

			for l in model.layers.itervalues() :
				try :
					l.learningScenario.update(self)
				except AttributeError :
					pass

			self.store["runInfos"]["epoch"] += 1