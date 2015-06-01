import theano
import theano.tensor as tt

class Cost_ABC(object) :
	"""This allows to create custom costs by adding stuff such as regularizations"""

	def __init__(self, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.hyperParameters = []

	def _costFct(self, targets, outputs) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

	def getCost(self, outputLayer) :
		return self._costFct(outputLayer.target, outputLayer.outputs)

class Null(Cost_ABC) :
	def _costFct(self, targets, outputs) :
		return outputs*0 + targets*0

class NegativeLogLikelihood(Cost_ABC) :
	"""cost fct for a probalistic intended to be used with a softmax output"""
	def _costFct(self, targets, outputs) :
		cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
		return cost

class MeanSquaredError(Cost_ABC) :
	"""The all time classic"""
	def _costFct(self, targets, outputs) :
		cost = tt.mean((outputs - targets) ** 2)
		return cost

class CategoricalCrossEntropy(Cost_ABC) :
	"""Returns the average number of bits needed to identify an event."""
	def _costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.categorical_crossentropy(outputs, targets) )
		return cost
		
class CrossEntropy(Cost_ABC) :
	"""Returns the average number of bits needed to identify an event."""
	def _costFct(self, targets, outputs) :
		cost = - tt.sum(outputs * tt.log(targets) + (1 - outputs) * tt.log(1 - targets), axis=1)
		return tt.mean(cost)

class BinaryCrossEntropy(Cost_ABC) :
	"""Use this one for binary data"""
	def _costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
		return cost