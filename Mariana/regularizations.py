__all__ = ["SingleLayerRegularizer_ABC", "L1", "L2"]

class SingleLayerRegularizer_ABC(object) :
	"""An abstract regularization to be applied to a layer."""

	def __init__(self, factor, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.factor = factor
		self.hyperparameters = ["factor"]

	def getFormula(self, layer) :
		"""Returns the expression to be added to the cost"""
		raise NotImplemented("Must be implemented in child")

class L1(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost
	.. math::

			factor * abs(Weights)
	"""

	def getFormula(self, layer) :
		# print "ssss L1" 
		return self.factor * ( abs(layer.W).sum() )

class L2(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost
	.. math::

			factor * (Weights)^2
	"""

	def getFormula(self, layer) :
		# print "pppp L2" 
		return self.factor * ( (layer.W * layer.W).sum() )
