import theano
import theano.tensor as tt

__all__ = ["Cost_ABC", "Null", "NegativeLogLikelihood", "MeanSquaredError", "CrossEntropy", "CategoricalCrossEntropy", "BinaryCrossEntropy"]

class Cost_ABC(object) :
	"""This is the interface a Cost must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
 	this class must also include a list attribute **self.hyperParameters** containing the names all attributes that must be considered
 	as hyper-parameters."""

	def __init__(self, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.hyperParameters = []

	def costFct(self, targets, outputs) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

class Null(Cost_ABC) :
	"""No cost at all"""
	def costFct(self, targets, outputs) :
		return outputs*0 + targets*0

class NegativeLogLikelihood(Cost_ABC) :
	"""For a probalistic output, works great with a softmax output layer"""
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
	"""Returns the average number of bits separating the the target form the output."""
	def costFct(self, targets, outputs) :
		cost = - tt.sum(outputs * tt.log(targets) + (1 - outputs) * tt.log(1 - targets), axis=1)
		return tt.mean(cost)

class BinaryCrossEntropy(Cost_ABC) :
	"""Use this one for binary data"""
	def costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
		return cost