import theano.tensor as tt

__all__ = ["Activation_ABC", "Pass", "Sigmoid", "Tanh", "Input", "ReLU", "Softmax"]

class Activation_ABC(object):
	"""All activations must inherit fron this class"""
	def __init__(self, *args, **kwargs):
		self.hyperParameters = []
		self.name = self.__class__.__name__

	def function(self, x) :
		"""the actual activation function that will be applied to the neurones."""
		raise NotImplemented("Must be implemented in child")

class Pass(Activation_ABC):
	"""
	simply returns x
	"""
	def __init__(self):
		Activation_ABC.__init__(self)
		
	def function(self, x):
		return x

class Sigmoid(Activation_ABC):
	"""
	.. math::

		1/ (1/ + exp(-x))"""
	def __init__(self):
		Activation_ABC.__init__(self)
		
	def function(self, x):
		return tt.nnet.sigmoid(x)

class Tanh(Activation_ABC):
	"""
	.. math::

		tanh(x)"""
	def __init__(self):
		Activation_ABC.__init__(self)

	def function(self, x):
		return tt.tanh(x)

class ReLU(Activation_ABC):
	"""
	.. math::

		max(0, x)"""
	def __init__(self):
		Activation_ABC.__init__(self)
				
	def function(self, x):
		#do not replace by theano's relu. It works bad with nets that have multiple outputs
		return tt.maximum(0., x)

class Softmax(Activation_ABC):
	"""Softmax to get a probabilistic output
	.. math::

		exp(x_i/T)/ sum_k( exp(x_k/T) )
	"""
	def __init__(self, temperature = 1):
		Activation_ABC.__init__(self)
		self.hyperParameters = ["temperature"]
		self.temperature = temperature

	def function(self, x):
		return tt.nnet.softmax(x/self.temperature)