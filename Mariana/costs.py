import theano.tensor as tt

# def crossEntropy(targets, outputs) :
# 	"""Use this one for binary data"""
# 	cost = -tt.nnet.binary_crossentropy(targets, outputs).mean()
# 	return cost

class Cost_ABC(object) :
	"""This allows to create custom costs by adding stuff such as regularizations"""

	def __init__(self, *args, **kwargs) :
		self.name = self.__class__.__name__
		# self.hyperParameters = []

	def _costFct(self, targets, outputs) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

	def getCost(self, outputLayer) :
		return self._costFct(outputLayer.target, outputLayer.outputs)

class Null(Cost_ABC) :
	def _costFct(self, targets, outputs) :
		return 0

class NegativeLogLikelihood(Cost_ABC) :
	"""cost fct for softmax"""
	def _costFct(self, targets, outputs) :
		cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
		return cost

class MeanSquaredError(Cost_ABC) :
	"""The all time classic"""
	def _costFct(self, targets, outputs) :
		cost = tt.mean((outputs - targets) ** 2)
		return cost