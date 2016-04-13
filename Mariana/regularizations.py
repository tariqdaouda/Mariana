from Mariana.abstraction import Abstraction_ABC

__all__ = ["SingleLayerRegularizer_ABC", "L1", "L2", "ActivationL1"]

class SingleLayerRegularizer_ABC(Abstraction_ABC) :
	"""An abstract regularization to be applied to a layer."""

	def apply(self, layer) :
		"""Apply to a layer and update networks's log"""
		hyps = {}
		for k in self.hyperParameters :
			hyps[k] = getattr(self, k)

		message = "%s uses %s regularization" % (layer.name, self.__class__.__name__)
		layer.network.logLayerEvent(layer, message, hyps)
		return self.getFormula(layer, x)

	def getFormula(self, layer) :
		"""Returns the expression to be added to the cost"""
		raise NotImplemented("Must be implemented in child")

class L1(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost. Weights will tend towards 0
	resulting in sparser weight matrices.
	.. math::

			factor * abs(Weights)
	"""
	def __init__(self, factor) :
		SingleLayerRegularizer_ABC.__init__(self)
		self.factor = factor
		self.hyperParameters = ["factor"]

	def getFormula(self, layer) :
		return self.factor * ( abs(layer.W).sum() )

class L2(SingleLayerRegularizer_ABC) :
	"""
	Will add this to the cost. Causes the weights to stay small
	.. math::

			factor * (Weights)^2
	"""
	def __init__(self, factor) :
		SingleLayerRegularizer_ABC.__init__(self)
		self.factor = factor
		self.hyperParameters = ["factor"]

	def getFormula(self, layer) :
		return self.factor * ( (layer.W * layer.W).sum() )

class ActivationL1(SingleLayerRegularizer_ABC) :
	"""
	L1 on the activations. Neurone activations will tend towards
	0, resulting into sparser representations.

	Will add this to the cost
	.. math::

			factor * abs(activations)
	"""
	def __init__(self, factor) :
		SingleLayerRegularizer_ABC.__init__(self)
		self.factor = factor
		self.hyperParameters = ["factor"]

	def getFormula(self, layer) :
		return self.factor * ( abs(layer.outputs).sum() )