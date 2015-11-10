import sys, os
from pyGeno.tools.parsers.CSVTools import CSVFile

__all__ = ["Recorder_ABC", "GGPlot2"]

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
 	"""This training recorder will create a nice CSV file fit for using with ggplot2 and will update
 	it as the training goes. It will also save the best model for each set of the trainer, and print
 	regular reports on the console.

 	:param int printRate: The rate at which the status is printed on the console. If set to <= to 0, will never print.
 	:param int write: The rate at which the status is written on disk
 	"""

 	def __init__(self, filename, printRate=1, writeRate=1):
		
		self.filename = filename.replace(".csv", "") + ".ggplot2.csv"
	
 		self.bestScores = {}
		self.currentScores = {}

		self.csvLegend = None
		self.csvFile = None

		self.length = 0

		self.printRate = printRate
		self.writeRate = writeRate

	def commit(self, store, model) :
		"""Appends the current state of the store to the CSV. This one is meant to be called by the trainer"""
		def _fillLine(csvFile, score, bestScore, setName, setLen, outputName, **csvValues) :
			line = csvFile.newLine()
			for k, v in csvValues.iteritems() :
				line[k] = v
			line["score"] = score
			line["best_score"] = bestScore[0]
			line["best_score_commit"] = bestScore[1]
			line["set"] = "%s(%s)" %(setName, setLen)
			line["output"] = outputName
			line.commit()
		
		self.length += 1
		if self.csvLegend is None :
			self.csvLegend = store["hyperParameters"].keys()
			self.csvLegend.extend(store["runInfos"].keys())
			self.csvLegend.extend( ["score", "best_score", "best_score_commit", "set", "output"] )

			self.csvFile = CSVFile(legend = self.csvLegend)
			self.csvFile.streamToFile( self.filename, writeRate = self.writeRate )

		for theSet, scores in store["scores"].iteritems() :
			self.currentScores[theSet] = {}
			if theSet not in self.bestScores :
				self.bestScores[theSet] = {}
			for outputName, score in scores.iteritems() :
				self.currentScores[theSet][outputName] = score
				if outputName not in self.bestScores[theSet] or score < self.bestScores[theSet][outputName][0] :
					self.bestScores[theSet][outputName] = (score, self.length)
					model.save("best-%s-%s-%s" % (outputName, theSet, self.filename))

				muchData = store["hyperParameters"]
				muchData.update(store["runInfos"]) 
				_fillLine(
					self.csvFile,
					self.currentScores[theSet][outputName],
					self.bestScores[theSet][outputName],
					theSet,
					store["setSizes"][theSet],
					outputName,
					**muchData
				)
	
		if self.printRate > 0 and (self.length%self.printRate) == 0:
			self.printCurrentState()

	def printCurrentState(self) :
		"""prints the current state stored in the recorder"""
		if self.length > 0 :
			print "\n==>rec: ggplot2, commit %s, pid: %s:" % (self.length, os.getpid())
			for setName, scores in self.bestScores.iteritems() :
				print "  |-%s set" % setName
				for outputName in scores :
					if self.currentScores[setName][outputName] == self.bestScores[setName][outputName][0] :
						highlight = "+best+"
					else :
						score, epoch = self.bestScores[setName][outputName]
						highlight = "(best: %s @ commit: %s)" % (score, epoch)
					
					print "    |->%s: %s %s" % (outputName, self.currentScores[setName][outputName], highlight)
		else :
			print "==>rec: ggplot2, nothing to show yet"
		
		sys.stdout.flush()

	def __len__(self) :
		"""returns the number of commits performed"""
		return self.length
