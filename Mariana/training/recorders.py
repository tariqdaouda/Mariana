import sys, os, types
from pyGeno.tools.parsers.CSVTools import CSVFile

__all__ = ["Recorder_ABC", "GGPlot2", "SavingRule_ABC", "SaveMin", "SaveMax", "Scores"]

class Scores(object) :
	"""Manage and store the scores returned by the trainer. This one is meant to be used internally by trainers."""
	def __init__(self) :
		self.reset()

	def reset(self):
		"""resets the store as if nothing ever happened"""
		self.currentScores = {}
		self.minScores = {}
		self.maxScores = {}
		self.epoch = 0

	def update(self, trainerScores, epoch) :
		"""update the with new scores"""
		def _rec(minScores, maxScores, dct, epoch, keys = []) :
			for k, v in dct.iteritems() :
				if type(v) is types.DictType :
					keys.append(k)
					if k not in minScores :
						minScores[k] = {}
						maxScores[k] = {}

					_rec(minScores[k], maxScores[k], v, epoch, keys = keys)
				else :
					try :
						if v < minScores[k][0] :
	 						minScores[k] = (v, epoch)
	 					elif v > maxScores[k][0] :
	 						maxScores[k] = (v, epoch)
	 				except KeyError :
						minScores[k] = (v, epoch)
 						maxScores[k] = (v, epoch)
	 	
	 	self.epoch = epoch
		self.currentScores = trainerScores
		_rec(self.minScores, self.maxScores, self.currentScores, epoch)

	def getScore(self, mapName, outputName, functionName) :
		"""return the last score for a map defined in the trainer (a set) an outpur layer and function of that output"""
		return (self.currentScores[mapName][outputName][functionName], self.epoch)

	def getMinScore(self, mapName, outputName, functionName) :
		"""return the min score acheived for a map defined in the trainer (a set) an outpur layer and function of that output"""
		return self.minScores[mapName][outputName][functionName]

	def getMaxScore(self, mapName, outputName, functionName) :
		"""return the max score acheived for a map defined in the trainer (a set) an outpur layer and function of that output"""
		return self.maxScores[mapName][outputName][functionName]

class SavingRule_ABC(object):
	"""Abstraction for saving rules"""

	def __init__(self, epochStart=0, savePeriod=1) :
		self.epochStart = epochStart
		self.savePeriod = savePeriod

		self.currentSavePeriod = 0

	def _shouldISave(self, recorder) :
		"""This is the function that is called by the recorder. If sets self.recorder to 'recorder' and then calls self.shouldISave()"""
		self.recorder = recorder

		if recorder.epoch >= self.epochStart :
			if not self.shouldISave(recorder) :
				return False

			self.currentSavePeriod += 1
			if (self.currentSavePeriod % self.savePeriod == 0) :
				return True
			else :
				return False
		else :
			return False

	def shouldISave(self, recorder) :
		"""The function that defines when a save should be performed"""
		raise NotImplemented("Should be implemented in child")

	def getFilename(self, recorder) :
		"""return the filename of the file to be saved"""
		raise NotImplemented("Should be implemented in child")

	def loadLast(self):
		"""load the last saved model"""
		import Mariana.network as MNET
		return MNET.loadModel(self.getFilename(self.recorder))

	def __repr__(self) :
		return "%s on %s" %(self.__class__.__name__, (self.mapName, self.outputName, self.functionName) )

class SaveMin(SavingRule_ABC) :
	"""Save the model when a new min value is reached

	:param string mapName: The set name of a map defined in the trainer usually something like "test" or "validation"
	:param string outputName: The name of the output layer to consider (you can also give the layer object)
	:param string functionName: The function of the output layer to consider usually "test".
	"""
	def __init__(self, mapName, outputName, functionName, *args, **kwargs) :
		super(SaveMin, self).__init__(*args, **kwargs)
		self.mapName = mapName
		if type(outputName) is types.StringType :
			self.outputName = outputName
		else :
			self.outputName = outputName.name

		self.functionName = functionName
		self.recorder = None

	def shouldISave(self, recorder) :
		s = recorder.scores.getScore(self.mapName, self.outputName, self.functionName)
		m = recorder.scores.getMinScore(self.mapName, self.outputName, self.functionName)
		if s[0] == m[0] :
			return True
		return False

	def getFilename(self, recorder) :
		return "bestMin-%s-%s-%s-%s" % (self.mapName, self.outputName, self.functionName, recorder.filename)

class SaveMax(SavingRule_ABC) :
	"""Save the model when a new max value is reached

	:param string mapName: The set name of a map defined in the trainer usually something like "test" or "validation"
	:param string outputName: The name of the output layer to consider (you can also give the layer object)
	:param string functionName: The function of the output layer to consider usually "test".
	"""

	def __init__(self, mapName, outputName, functionName, *args, **kwargs) :
		super(SaveMax, self).__init__(*args, **kwargs)
		self.mapName = mapName
		if type(outputName) is types.StringType :
			self.outputName = outputName
		else :
			self.outputName = outputName.name

		self.functionName = functionName
		self.recorder = None
	def shouldISave(self, recorder) :
		s = recorder.scores.getScore(self.mapName, self.outputName, self.functionName)
		m = recorder.scores.getMaxScore(self.mapName, self.outputName, self.functionName)
		if s[0] == m[0] :
			return True
		return False

	def getFilename(self, recorder) :
		return "bestMax-%s-%s-%s-%s" % (self.mapName, self.outputName, self.functionName, recorder.filename)

class SavePeriod(SavingRule_ABC) :
	"""Periodically saves the current model
	
	:param boolean distinct: If False, each new save will overwrite the previous one.
	"""
	def __init__(self, period, distinct, *args, **kwargs) :
		super(SavePeriod, self).__init__(savePeriod=period, **kwargs)
		self.distinct = distinct
	
	def shouldISave(self, recorder) :
		return True
		# return (recorder.epoch % self.period) == 0

	def getFilename(self, recorder) :
		if self.distinct :
			return "periodicallySaved-epoch_%s-%s" % (recorder.epoch, recorder.filename)
		else :
			return "periodicallySaved-%s" % (recorder.filename)

class Recorder_ABC(object) :
	"""A recorder is meant to be plugged into a trainer to record the
	advancement of the training. This is the interface a Recorder must expose."""
	
	def commit(self, store, model) :
		"""Does something with the currenty state of the trainer's store and the model"""
		raise NotImplemented("Should be implemented in child")

	def __len__(self) :
		"""returns the number of commits performed"""
		raise NotImplemented("Should be implemented in child")

class GGPlot2(Recorder_ABC):
 	"""This training recorder will create a nice TSV (tab delimited) file fit for using with ggplot2 and will update
 	it as the training goes. It will also save the best model for each set of the trainer, and print
 	regular reports on the console.

 	:param string filename: The filename of the tsv to be generated. the extension '.ggplot2.tsv' will be added automatically
 	:param list whenToSave: List of saving rules.
 	:param int printRate: The rate at which the status is printed on the console. If set to <= to 0, will never print.
 	:param int write: The rate at which the status is written on disk
 	"""

 	def __init__(self, filename, whenToSave = [], printRate=1, writeRate=1):
		
		self.filename = filename.replace(".tsv", "") + ".ggplot2.tsv"
		self.scores = Scores()

		self.csvLegend = None
		self.csvFile = None

		self.length = 0
		self.epoch = 0

		self.trainerStore = None

		self.printRate = printRate
		self.writeRate = writeRate
		self.whenToSave = whenToSave

	def commit(self, store, model) :
		"""Appends the current state of the store to the CSV. This one is meant to be called by the trainer"""
		def _fillLine(csvFile, score, minScore, maxScore, mapName, setLen, outputName, outputFunction, **csvValues) :
			line = csvFile.newLine()

			for k, v in csvValues.iteritems() :
				line[k] = v
			line["score"] = score[0]
			line["min_score"] = minScore[0]
			line["min_score_commit"] = minScore[1]
			
			line["max_score"] = maxScore[0]
			line["max_score_commit"] = maxScore[1]
			
			line["set"] = "%s" %(mapName)
			line["set_size"] = "%s" %(setLen)
			line["output_layer"] = outputName
			line["output_function"] = outputFunction
			line.commit()
		
		self.length += 1
		if self.csvLegend is None :
			self.csvLegend = store["hyperParameters"].keys()
			self.csvLegend.extend(store["runInfos"].keys())
			self.csvLegend.extend( ["score", "min_score", "min_score_commit", "max_score", "max_score_commit", "set", "set_size", "output_layer", "output_function"] )

			self.csvFile = CSVFile(legend = self.csvLegend, separator = "\t")
			self.csvFile.streamToFile( self.filename, writeRate = self.writeRate )

		muchData = store["hyperParameters"]
		muchData.update(store["runInfos"]) 

		self.scores.update(store["scores"], store["runInfos"]["epoch"])
		for mapName, os in store["scores"].iteritems() :
			for outputName, fs in os.iteritems() :
				for functionName in fs :
					_fillLine(
						self.csvFile,
						self.scores.getScore(mapName, outputName, functionName),
						self.scores.getMinScore(mapName, outputName, functionName),
						self.scores.getMaxScore(mapName, outputName, functionName),
						mapName,
						store["setSizes"][mapName],
						outputName,
						functionName,
						**muchData
					)

		self.trainerStore = store
		self.epoch = store["runInfos"]["epoch"]
		self.runtime_min = store["runInfos"]["runtime_min"]
		if self.printRate > 0 and (self.length%self.printRate) == 0:
			self.printCurrentState()

		for w in self.whenToSave :
			if w._shouldISave(self) :
				model.save(w.getFilename(self))

	def printCurrentState(self) :
		"""prints the current state stored in the recorder"""
		if self.length > 0 :
			print "\n==>rec: ggplot2, epoch %s, runtime %s(mins) commit %s, pid: %s:" % (self.epoch, self.runtime_min, self.length, os.getpid())
			for mapName, outs in self.scores.currentScores.iteritems() :
				print "  |-%s set" % mapName
				for outputName, fs in outs.iteritems() :
					print "    |-%s" % outputName
					for functionName in fs :
						s = self.scores.getScore(mapName, outputName, functionName)
						mi = self.scores.getMinScore(mapName, outputName, functionName)
						ma = self.scores.getMaxScore(mapName, outputName, functionName)

						highlight = []
						if s[0] == mi[0] :
							highlight.append("+min+")
						else :
							highlight.append("%s@%s" % (mi[0], mi[1]))

						if s[0] == ma[0] :
							highlight.append("+max+")
						else :
							highlight.append("%s@%s" % (ma[0], ma[1]))

						print "      |-%s: %s [%s]" % (functionName, s[0], "; ".join(highlight))
		else :
			print "==>rec: ggplot2, nothing to show yet"
		
		sys.stdout.flush()

	def __repr__(self):
		return "<recorder: %s, filename: %s>" % (self.__class__.__name__, self.filename)

	def __len__(self) :
		"""returns the number of commits performed"""
		return self.length
