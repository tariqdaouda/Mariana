import theano
import theano.tensor as tt

__all__ = ["Cost_ABC", "Null", "NegativeLogLikelihood", "MeanSquaredError", "CategoricalCrossEntropy", "CrossEntropy", "BinaryCrossEntropy"]

class Cost_ABC(object) :
	"""This is the interface a Cost must expose"""

	def __init__(self, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.hyperParameters = []

	def costFct(self, targets, outputs) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

	def getCost(self, outputLayer) :
		return self.costFct(outputLayer.target, outputLayer.outputs)

class Null(Cost_ABC) :
	"""No cost at all"""
	def costFct(self, targets, outputs) :
		return outputs*0 + targets*0

class NegativeLogLikelihood(Cost_ABC) :
	"""cost fct for a probalistic intended to be used with a softmax output"""
	def costFct(self, targets, outputs) :
		cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
		return cost

class MeanSquaredError(Cost_ABC) :
	"""The all time classic"""
	def costFct(self, targets, outputs) :
		cost = tt.mean((outputs - targets) ** 2)
		return cost

class CategoricalCrossEntropy(Cost_ABC) :
	"""Returns the average number of bits needed to identify an event."""
	def costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.categorical_crossentropy(outputs, targets) )
		return cost
		
class CrossEntropy(Cost_ABC) :
	"""Returns the average number of bits needed to identify an event."""
	def costFct(self, targets, outputs) :
		cost = - tt.sum(outputs * tt.log(targets) + (1 - outputs) * tt.log(1 - targets), axis=1)
		return tt.mean(cost)

class BinaryCrossEntropy(Cost_ABC) :
	"""Use this one for binary data"""
	def costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
		return cost