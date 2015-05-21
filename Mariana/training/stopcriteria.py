
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

class ScoreWall(StopCriterion_ABC) :
	"""Stops training when a givven score is reached"""
	def __init__(self, wallValue, theSet, outputLayer = None) :
		"""if outputLayer is None, will consider the average of all outputs"""
		StopCriterion_ABC.__init__(self)

		self.theSet = theSet
		self.outputLayer = outputLayer
		self.wallValue = wallValue

	def stop(self, trainer) :
		if self.outputLayer is None :
			curr = trainer.recorder.getAverageCurrentScore(self.theSet)
		else :
			curr = trainer.recorder.getCurrentScore(self.outputLayer, self.theSet)
	
		if curr <= self.wallValue :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached score wall %s" % self.wallValue

class GeometricEarlyStopping(StopCriterion_ABC) :
	"""Geometrically increases the patiences with the epochs and stops the training when the patience is over."""
	def __init__(self, theSet, patience, patienceIncreaseFactor, significantImprovement, outputLayer = None) :
		"""if outputLayer is None, will consider the average of all outputs"""
		StopCriterion_ABC.__init__(self)
		
		self.outputLayer = outputLayer
		self.theSet = theSet
		self.patience = patience
		self.patienceIncreaseFactor = patienceIncreaseFactor
		self.wall = patience
		self.significantImprovement = significantImprovement

		self.bestScore = None

	def stop(self, trainer) :
		if self.wall <= 0 :
			return True

		if self.outputLayer is None :
			curr = trainer.store["scores"][self.theSet]["average"]
		else :
			curr = trainer.store["scores"][self.theSet][self.outputLayer.name]
			
		if self.bestScore is None :
			self.bestScore = curr
		elif curr < (self.bestScore + self.significantImprovement) :
			self.bestScore = curr
		
		
			self.wall = max(self.patience, trainer.currentEpoch * self.patienceIncreaseFactor)
	
		self.wall -= 1	
		return False
	
	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Early stopping, no patience left"
