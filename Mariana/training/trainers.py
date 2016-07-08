import cPickle, time, sys, os, traceback, types, json, signal
import numpy
from collections import OrderedDict
import theano.tensor as tt

import Mariana.network as MNET
import Mariana.training.recorders as MREC
import Mariana.training.stopcriteria as MSTOP
import Mariana.candies as MCAN
import Mariana.settings as MSET

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

		import simplejson, signal, cPickle

		def _handler_sig_term(sig, frame) :
			_dieGracefully("SIGTERM", None)
			sys.exit(sig)

		def _dieGracefully(exType, tb = None) :
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
			sstore = str(self.store).replace("'", '"').replace("True", 'true').replace("False", 'false')
			try :
				f.write(
					"store:\n%s" % json.dumps(
						json.loads(sstore), sort_keys=True,
						indent=4,
						separators=(',', ': ')
					)
				)
			except Exception as e:
				print "Warning: Couldn't format the store to json, saving it ugly."
				print "Reason:", e
				f.write(sstore)

			if tb is not None :
				f.write("\nTraceback\n---------\n")
				f.write(str(traceback.extract_tb(tb)).replace("), (", "),\n(").replace("[(","[\n(").replace(")]",")\n]"))
			f.flush()
			f.close()
			f = open(filename + ".store.pkl", "wb")
			cPickle.dump(self.store, f)
			f.close()

		signal.signal(signal.SIGTERM, _handler_sig_term)
		if MSET.VERBOSE :
			print "\n" + "Training starts."		
		MCAN.friendly("Process id", "The pid of this run is: %d" % os.getpid())

		if recorder == "default" :
			params = {
				"printRate" : 1,
				"writeRate" : 1
			}
			recorder = MREC.GGPlot2(runName, **params)
			MCAN.friendly(
				"Default recorder",
				"The trainer will recruit the default 'GGPlot2' recorder with the following arguments:\n\t %s.\nResults will be saved into '%s'." % (params, recorder.filename)
			)
	
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
			sstore = str(self.store).replace("'", '"').replace("True", 'true').replace("False", 'false')
			try :
				f.write(
					"store:\n%s" % json.dumps(
						json.loads(sstore), sort_keys=True,
						indent=4,
						separators=(',', ': ')
					)
				)
			except Exception as e:
				print "Warning: Couldn't format the store to json, saving it ugly."
				print "Reason:", e
				f.write(sstore)

			f.flush()
			f.close()
			model.save(filename)
			f = open(filename + ".store.pkl", "wb")
			cPickle.dump(self.store, f)
			f.close()

		except KeyboardInterrupt :
			if not self.saveIfMurdered :
				raise
			exType, ex, tb = sys.exc_info()
			_dieGracefully(exType, tb)
			raise
		except :
			if not self.saveIfMurdered :
				raise
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
	RANDOM_PICK_TRAINING = 2

	ALL_SET = -1

	def __init__(self,
		trainMaps,
		testMaps,
		validationMaps,
		trainMiniBatchSize,
		stopCriteria = [],
		testMiniBatchSize = -1,
		validationMiniBatchSize = -1,
		saveIfMurdered = True,
		trainFunctionName = "train",
		testFunctionName="test",
		validationFunctionName = "test") :
		"""
			:param DatasetMaps trainMaps: Layer mappings for the training set
			:param DatasetMaps testtrainMaps: Layer mappings for the testing set
			:param DatasetMaps validationMaps: Layer mappings for the validation set, if you do not wich to set one, pass None as argument
			:param int trainMiniBatchSize: The size of a training minibatch, use DefaultTrainer.ALL_SET for the whole set
			:param list stopCriteria: List of StopCriterion objects 
			:param int testMiniBatchSize: The size of a testing minibatch, use DefaultTrainer.ALL_SET for the whole set
			:param int validationMiniBatchSize: The size of a validationMiniBatchSize minibatch
			:param bool saveIfMurdered: Die gracefully in case of Exception or SIGTERM and save the current state of the model and logs
			:param string trainFunctionName: The name of the function to use for training
			:param string testFunctionName: The name of the function to use for testing
			:param string validationFunctionName: The name of the function to use for testing in validation
		"""
		
		Trainer_ABC.__init__(self)

		self.maps = {
			"train": trainMaps,
			"test": testMaps,
		}

		if validationMaps is not None :
			self.maps["validation"] = validationMaps

		self.miniBatchSizes = {
			"train" : trainMiniBatchSize,
			"test" : testMiniBatchSize,
			"validation" : validationMiniBatchSize
		}

		self.stopCriteria = stopCriteria		
		self.saveIfMurdered = saveIfMurdered
		
		self.trainingOrdersHR = {
			self.SIMULTANEOUS_TRAINING : "SIMULTANEOUS",
			self.SEQUENTIAL_TRAINING : "SEQUENTIAL",
			self.RANDOM_PICK_TRAINING : "RANDOM_PICK"
		}

		self.trainFunctionName = trainFunctionName
		self.testFunctionName = testFunctionName
		self.validationFunctionName = validationFunctionName

	def start(self, runName, model, recorder = "default", trainingOrder = 0, moreHyperParameters={}) :
		"""starts the training, cf. run() for the a description of the arguments"""
		Trainer_ABC.start( self, runName, model, recorder, trainingOrder, moreHyperParameters )

	def run(self, name, model, recorder, trainingOrder, moreHyperParameters) :
		"""
			:param str runName: The name of this run
			:param Recorder recorder: A recorder object
			:param int trainingOrder:
				* DefaultTrainer.SEQUENTIAL_TRAINING: Each output will be trained indipendetly on it's own epoch
				* DefaultTrainer.SIMULTANEOUS_TRAINING: All outputs are trained within the same epoch with the same inputs
				* Both are in O(m*n), where m is the number of mini batches and n the number of outputs
				* DefaultTrainer.RANDOM_PICK_TRAINING: Will pick one of the outputs at random for each example

			:param bool reset: Should the trainer be reset before starting the run
			:param dict moreHyperParameters: If provided, the fields in this dictionary will be included into the log .csv file
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

		def _trainTest(aMap, modelFct, trainingOrder, miniBatchSize, inputLayers, outputLayers) :
			scores = {}
			layerList = inputLayers
			if miniBatchSize == DefaultTrainer.ALL_SET :
				for output in outputLayers :
					layerList.append(output)
					kwargs = dict( aMap.getAll(layerList = layerList) )
					res = modelFct(output, **kwargs)
					scores[output.name] = dict(res)
					layerList.pop(-1)
			else :
				if trainingOrder == DefaultTrainer.SEQUENTIAL_TRAINING :
					for output in outputLayers :
						for i in xrange(0, len(aMap), miniBatchSize) :
							layerList.append(output)
							batchData = aMap.getBatch(i, miniBatchSize, layerList = layerList)
							res = modelFct(output, **batchData)
							if output.name not in scores :
								scores[output.name] = {}
							
							for k, v in res.iteritems() :
								try :
									scores[output.name][k].append(v)
								except KeyError:
									scores[output.name][k] = [v]
							
							layerList.pop(-1)
				elif trainingOrder == DefaultTrainer.SIMULTANEOUS_TRAINING :
					for i in xrange(0, len(aMap), miniBatchSize) :
						batchData = aMap.getBatch(i, miniBatchSize, layerList = layerList)
						for output in outputLayers :
							layerList.append(output)
							res = modelFct(output, **batchData)
							
							if output.name not in scores :
								scores[output.name] = {}
							
							for k, v in res.iteritems() :
								try :
									scores[output.name][k].append(v)
								except KeyError:
									scores[output.name][k] = [v]
							
							layerList.pop(-1)
				
				elif trainingOrder == DefaultTrainer.RANDOM_PICK_TRAINING :
					outputOrder = list(numpy.random.randint(0, len(outputLayers), len(aMap)/miniBatchSize +1))
					for i in xrange(0, len(aMap), miniBatchSize):
						output = outputLayers[outputOrder.pop()]
						layerList.append(output)
						batchData = aMap.getBatch(i, miniBatchSize, layerList = layerList)
						res = modelFct(output, **batchData)
					
						if output.name not in scores :
							scores[output.name] = {}
						
						for k, v in res.iteritems() :
							try :
								scores[output.name][k].append(v)
							except KeyError:
								scores[output.name][k] = [v]
						
						layerList.pop(-1)
				else :
					raise ValueError("Unknown training order: %s" % trainingOrder)

			if len(outputLayers) > 1 :
				keys = {}
	
			for outputName in scores :
				for fname, v in scores[outputName].iteritems() :
					avg = numpy.mean(v)
					scores[outputName][fname] = avg
					if len(outputLayers) > 1 :
						try :
							keys[fname].append(avg)
						except KeyError :
							keys[fname] = [avg]

			if len(outputLayers) > 1 :
				for k, v in keys.itervalues() :
					scores["average"] = numpy.mean(v)
			
			return scores

		if trainingOrder not in self.trainingOrdersHR:
			raise ValueError("Unknown training order: %s" % trainingOrder)

		legend = ["name", "epoch", "runtime(min)", "training_order"]
		legend.extend(moreHyperParameters.keys())
		hyperParameters = OrderedDict()
		for l in model.layers.itervalues() :
			hyperParameters["%s_shape" % l.name] = list(l.getOutputShape()) #cast to list for json save
			try :
				hyperParameters["activation"] = l.activation.__name__
			except AttributeError :
				pass
			setHPs(l, "learningScenario", hyperParameters)
			setHPs(l, "decorators", hyperParameters)
			setHPs(l, "activation", hyperParameters)
			if l.type == MNET.TYPE_OUTPUT_LAYER :
				setHPs(l, "costObject", hyperParameters)

		legend.extend(hyperParameters.keys())
		self.store["hyperParameters"].update(hyperParameters)
		self.store["hyperParameters"].update(moreHyperParameters)
		
		self.store["runInfos"]["name"] = name
		self.store["runInfos"]["training_order"] = self.trainingOrdersHR[trainingOrder]
		self.store["runInfos"]["pid"] = os.getpid()
	
		self.recorder = recorder
		
		startTime = time.time()
		self.store["runInfos"]["epoch"] = 0
		inputLayers = model.inputs.values()
		outputLayers = model.outputs.values()

		for mapName in self.maps :
			self.store["setSizes"][mapName] = self.maps[mapName].getMinFullLength()
			self.store["scores"][mapName] = {}
			for o in outputLayers :
				self.store["scores"][mapName][o.name] = {}

		while True :
			for mapName in ["train", "test", "validation"] :
				try :
					aMap = self.maps[mapName]
					if len(aMap) > 0 :
						scores = {}
						aMap.commit()
						if mapName == "train" :
							modelFct = getattr(model, self.trainFunctionName)
						elif mapName == "test" :
							modelFct = getattr(model, self.testFunctionName)
						elif mapName == "validation" :
							modelFct = getattr(model, self.validationFunctionName)
						else :
							raise ValueError("Unknown map name: '%s'" % mapName)
						
						scores = _trainTest(
							aMap,
							modelFct,
							trainingOrder,
							self.miniBatchSizes[mapName],
							inputLayers,
							outputLayers
						)
						self.store["scores"][mapName] = scores
				except KeyError :
					pass
			
			runtime = (time.time() - startTime)/60
			self.store["runInfos"].update( (
				("runtime_min", runtime),
			) )
	
			self.recorder.commit(self.store, model)
			for crit in self.stopCriteria :
				if crit.stop(self) :
					raise MSTOP.EndOfTraining(crit)

			for l in model.layers.itervalues() :
				try :
					l.learningScenario.update(self.store)
				except AttributeError :
					pass

			self.store["runInfos"]["epoch"] += 1
	
	def runCustom(self, model, outputName, functionName, dataMapName, miniBatchSize = -1) :
		"""Runs a given function, of a given model, applied to a given output using one of dataset maps of the trainer.
		In order for this function to work, 'model' must either be the model on wich the dataset maps have been defined, or have layers
		that have the same names as those of the model on wich the dataset maps have been defined."""		
		
		modelFct = getattr(model, functionName)
		aMap = self.maps[dataMapName]

		layerList = [outputName]
		for l in model.inputs.itervalues() :
			layerList.append(l.name)

		if miniBatchSize == DefaultTrainer.ALL_SET :
			kwargs = dict( aMap.getAll(layerList = layerList) )
			res = modelFct(outputName, **kwargs)
		else :
			for i in xrange(0, len(aMap), miniBatchSize) :
				batchData = aMap.getBatch(i, miniBatchSize, layerList = layerList)
				res = modelFct(outputName, **batchData)
		return res[0]