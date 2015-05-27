import sys, os
from pyGeno.tools.parsers.CSVTools import CSVFile

class Recorder_ABC(object) :

	def commit(self, store, model) :
		"""Does something with the currenty state of the trainer's store and the model"""
		raise NotImplemented("Should be implemented in child")

	def __len__(self) :
		"""returns the number of commits performed"""
		raise NotImplemented("Should be implemented in child")

class GGPlot2(Recorder_ABC):
 	"""This training recorder will create a nice CSV file fit for using with ggplot2. It will also print regular
 	reports if you tell it to be verbose and save the best models"""
 	def __init__(self, filename, verbose = True):
		
		self.filename = filename.replace(".csv", "") + ".ggplot2.csv"
		self.verbose = verbose
	
 		self.bestScores = {}
		self.currentScores = {}

		self.csvLegend = None
		self.csvFile = None

		self.length = 0

	def commit(self, store, model) :
		"""Appends the current state of the store to the CSV """
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
			self.csvFile.streamToFile( self.filename, writeRate = 1 )

		for theSet, scores in store["scores"].iteritems() :
			self.currentScores[theSet] = {}
			if theSet not in self.bestScores :
				self.bestScores[theSet] = {}
			for outputName, score in scores.iteritems() :
				self.currentScores[theSet][outputName] = score
				if outputName not in self.bestScores[theSet] or score < self.bestScores[theSet][outputName][0] :
					self.bestScores[theSet][outputName] = (score, self.length)
					model.save("best-%s-%s" % (theSet, self.filename))

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
	
	
		if self.verbose :
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