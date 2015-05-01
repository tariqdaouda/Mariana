
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

class TestScoreWall(StopCriterion_ABC) :
	"""Stops training when maxEpochs is reached"""
	def __init__(self, wallValue) :
		StopCriterion_ABC.__init__(self)
		self.wallValue = wallValue

	def stop(self, trainer) :
		if trainer.currentTestScore <= self.wallValue :
			return True
		return False

	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Reached test score wall %s" % self.wallValue

class GeometricEarlyStopping(StopCriterion_ABC) :
	"""Geometrically increases the patiences with the epochs and stops the training when the patience is over."""
	def __init__(self, theSet, patience, patienceIncreaseFactor, significantImprovement) :
		"""theSet must either be 'test' or 'validation'"""
		
		StopCriterion_ABC.__init__(self)
		if theSet.lower() != "test" and theSet.lower() != "validation" :
			raise KeyError("theSet must either be 'test' or 'validation'")

		self.theSet = theSet.lower()
		self.patience = patience
		self.patienceIncreaseFactor = patienceIncreaseFactor
		self.wall = patience
		self.significantImprovement = significantImprovement

	def stop(self, trainer) :
		if self.wall <= 0 :
			return True

		if self.theSet == "test" :
			if trainer.currentTestScore < (trainer.bestTestScore + self.significantImprovement) :
				self.wall = max(self.patience, trainer.currentEpoch * self.patienceIncreaseFactor)
		else :
			if trainer.currentValidationScore < (trainer.bestValidationScore + self.significantImprovement) :
				self.wall = max(self.patience, trainer.currentEpoch * self.patienceIncreaseFactor)
		
		self.wall -= 1	
		return False
	
	def endMessage(self) :
		"""returns information about the reason why the training stopped"""
		return "Early stopping, no patience left"
