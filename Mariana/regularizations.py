__all__ = ["SingleLayerRegularizer_ABC", "L1", "L2"]

class SingleLayerRegularizer_ABC(object) :
	"""An abstract regularization to be applied to a layer."""

	def __init__(self, *args, **kwargs) :
		self.name = self.__class__.__name__

	def getFormula(self, layer) :
		"""Returns the expression to be added to the cost"""
		raise NotImplemented("Must be implemented in child")

class L1(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost
	.. math::

			factor * abs(Weights)
	"""
	def __init__(self, factor) :
		SingleLayerRegularizer_ABC.__init__(self)
		self.factor = factor
		self.hyperparameters = ["factor"]

	def getFormula(self, layer) :
		return self.factor * ( abs(layer.W).sum() )

class L2(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost
	.. math::

			factor * (Weights)^2
	"""
	def __init__(self, factor) :
		SingleLayerRegularizer_ABC.__init__(self)
		self.factor = factor
		self.hyperparameters = ["factor"]

	def getFormula(self, layer) :
		return self.factor * ( (layer.W * layer.W).sum() )
