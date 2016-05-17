import theano
import theano.tensor as tt
from Mariana.abstraction import Abstraction_ABC

__all__ = ["Cost_ABC", "Null", "NegativeLogLikelihood", "MeanSquaredError", "CrossEntropy", "CategoricalCrossEntropy", "BinaryCrossEntropy"]

class Cost_ABC(Abstraction_ABC) :
	"""This is the interface a Cost must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
 	this class must also include a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
 	as hyper-parameters."""

	def apply(self, layer, targets, outputs, purpose) :
		"""Apply to a layer and update networks's log. Purpose is supposed to be a sting such as 'train' or 'test'"""
		hyps = {}
		for k in self.hyperParameters :
			hyps[k] = getattr(self, k)

		message = "%s uses cost %s for %s" % (layer.name, self.__class__.__name__, purpose)
		layer.network.logLayerEvent(layer, message, hyps)
		return self.costFct(targets, outputs)

	def costFct(self, targets, outputs) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

class Null(Cost_ABC) :
	"""No cost at all"""
	def costFct(self, targets, outputs) :
		return tt.sum(outputs*0)

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
		
class CrossEntropy(CategoricalCrossEntropy) :
	"""Short hand for CategoricalCrossEntropy"""
	pass

class BinaryCrossEntropy(Cost_ABC) :
	"""Use this one for binary data"""
	def costFct(self, targets, outputs) :
		cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
		return cost