__all__ = ["EndOfTraining", "StopCriterion_ABC", "EpochWall", "GeometricEarlyStopping"]

class EndOfTraining(Exception) :
	"""Exception raised when a training criteria is met"""
	def __init__(self, stopCriterion) :
		self.stopCriterion = stopCriterion
		self.message = "End of training: %s" % stopCriterion.endMessage()

class StopCriterion_ABC(object) :
	"""This defines the interface that a StopCriterion must expose"""

	def __init__(self, *args, **kwrags) :
		self.name = self.__class__.__name__

	def stop(self, trainer) :
		"""The actual function that is called by the trainer at each epoch. Must be implemented in children"""
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
		if trainer.store["runInfos"]["epoch"] >= self.maxEpochs :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached epoch wall %s" % self.maxEpochs

class ScoreWall(StopCriterion_ABC) :
	"""Stops training when a given score is reached"""
	def __init__(self, wallValue, datasetMap, outputLayer = None) :
		"""if outputLayer is None, will consider the average of all outputs"""
		StopCriterion_ABC.__init__(self)

		self.datasetMap = datasetMap
		self.datasetName = None
		self.outputLayer = outputLayer
		self.wallValue = wallValue

	def stop(self, trainer) :
		
		if self.datasetName is None :
			found = False
			for name, m in trainer.maps.iteritems() :
				if m is self.datasetMap :
					self.datasetName = name
					found = True
					break
			if not found :
				raise ValueError("the trainer does not know the supplied dataset map")

		if self.outputLayer is None :
			curr = trainer.store["scores"][self.datasetName]["average"]
		else :
			curr = trainer.store["scores"][self.datasetName][self.outputLayer.name]
	
		if curr <= self.wallValue :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached score wall %s" % self.wallValue

class GeometricEarlyStopping(StopCriterion_ABC) :
	"""Geometrically increases the patiences with the epochs and stops the training when the patience is over."""
	def __init__(self, datasetMap, patience, patienceIncreaseFactor, significantImprovement, outputLayer = None) :
		"""if outputLayer is None, will consider the average of all outputs"""
		StopCriterion_ABC.__init__(self)
		
		self.outputLayer = outputLayer
		self.datasetMap = datasetMap
		self.datasetName = None
		self.patience = patience
		self.patienceIncreaseFactor = patienceIncreaseFactor
		self.wall = patience
		self.significantImprovement = significantImprovement

		self.bestScore = None

	def stop(self, trainer) :
		if self.wall <= 0 :
			return True

		if self.datasetName is None :
			found = False
			for name, m in trainer.maps.iteritems() :
				if m is self.datasetMap :
					self.datasetName = name
					found = True
					break
			if not found :
				raise ValueError("the trainer does not know the supplied dataset map")

		if self.outputLayer is None :
			curr = trainer.store["scores"][self.datasetName]["average"]
		else :
			curr = trainer.store["scores"][self.datasetName][self.outputLayer.name]
			
		if self.bestScore is None :
			self.bestScore = curr
		elif curr < (self.bestScore + self.significantImprovement) :
			self.bestScore = curr
		
		
			self.wall = max(self.patience, trainer.store["runInfos"]["epoch"] * self.patienceIncreaseFactor)
	
		self.wall -= 1	
		return False
	
	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Early stopping, no patience left"
