import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
from Mariana.abstraction import Abstraction_ABC
import Mariana.initializations as MI

__all__= ["Decorator_ABC", "OutputDecorator_ABC", "BatchNormalization", "Center", "Normalize", "Mask", "BinomialDropout", "WeightSparsity", "InputSparsity"]

class Decorator_ABC(Abstraction_ABC) :
	"""A decorator is a modifier that is applied on a layer."""

	def __call__(self, *args, **kwargs) :
		self.decorate(*args, **kwargs)

	def apply(self, layer) :
		"""Apply to a layer and update networks's log"""
		hyps = {}
		for k in self.hyperParameters :
			hyps[k] = getattr(self, k)

		message = "%s is decorated by %s" % (layer.name, self.__class__.__name__)
		layer.network.logLayerEvent(layer, message, hyps)
		return self.decorate(layer)

	def decorate(self, layer) :
		"""The function that all decorator_ABCs must implement"""
		raise NotImplemented("This one should be implemented in child")

class OutputDecorator_ABC(Decorator_ABC) :

	def __init__(self, trainOnly, *args, **kwargs) :
		"""
			Output decorators modify either layer.outputs or layer.test_outputs and are used to implement stuff
			such as noise injection.

			:param bool trainOnly: if True, the decoration will not be applied in a test context
			(ex: while calling *test()*, *propagate()*)
		"""
		Decorator_ABC.__init__(self, *args, **kwargs)
		self.hyperParameters.extend(["trainOnly"])
		self.trainOnly = trainOnly

class Mask(OutputDecorator_ABC):
	"""Applies a fixed mask to the outputs of the layer. This layers does this:

		.. math::

			outputs = outputs * mask

	If you want to remove some parts of the output use 0s, if you want to keep them as they are use 1s.
	Anything else will lower or increase the values by the given factors.

	:param array/list mask: It should have the same dimensions as the layer's outputs.

	"""
	def __init__(self, mask, trainOnly = False, *args, **kwargs):
		OutputDecorator_ABC.__init__(self, trainOnly, *args, **kwargs)
		self.mask = mask
		self.hyperParameters.extend([])

	def decorate(self, layer) :
		layer.outputs = layer.outputs * self.mask
		if not self.trainOnly :
			layer.testOutputs = layer.testOutputs * self.mask

class BinomialDropout(OutputDecorator_ABC):
	"""Stochastically mask some parts of the output of the layer. Use it to make things such as denoising autoencoders and dropout layers"""
	def __init__(self, ratio, trainOnly = True, *args, **kwargs):
		OutputDecorator_ABC.__init__(self, trainOnly, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1)
		self.ratio = ratio
		self.seed = MSET.RANDOM_SEED
		self.hyperParameters.extend(["ratio"])

	def _decorate(self, outputs) :
		rnd = tt.shared_randomstreams.RandomStreams()
		mask = rnd.binomial(n = 1, p = (1-self.ratio), size = outputs.shape)
		# cast to stay in GPU float limit
		mask = tt.cast(mask, theano.config.floatX)
		return (outputs * mask) #/ self.ratio

	def decorate(self, layer) :
		if self.ratio > 0 :
			layer.outputs = self._decorate(layer.outputs)
			if not self.trainOnly :
				layer.testOutputs = self._decorate(layer.testOutputs)

class Center(Decorator_ABC) :
	"""Centers the outputs by substracting the mean"""
	def decorate(self, layer) :
		layer.output = layer.output-tt.mean(layer.output)
		layer.testOutput = layer.testOutput-tt.mean(layer.testOutput)

class Normalize(Decorator_ABC) :
	"""
	Normalizes the outputs by substracting the mean and deviding by the standard deviation

	:param float espilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
	"""

	def __init__(self, espilon=1e-6) :
		Decorator_ABC.__init__(self)
		self.espilon = espilon

	def decorate(self, layer) :
		std = tt.sqrt( tt.var(layer.outputs) + self.espilon )
		layer.output = ( layer.output-tt.mean(layer.output) / std )
	
		std = tt.sqrt( tt.var(layer.testOutputs) + self.espilon )
		layer.testOutput = ( layer.testOutput-tt.mean(layer.testOutput) ) / std

class BatchNormalization(Decorator_ABC):
	"""Applies Batch Normalization to the outputs of the layer.
	Implementation according to Sergey Ioffe and Christian Szegedy (http://arxiv.org/abs/1502.03167)
	
		.. math::

			W * ( inputs - mean(mu) )/( std(inputs) ) + b

		Where W and b are learned and std stands for the standard deviation. The mean and the std are computed accross the whole minibatch.

		:param float epsilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
		:param initialization WInitialization: How to initizalise the weights. This decorator is smart enough to use layer initializations.
		:param initialization bInitialization: Same for bias

	"""

	def __init__(self, WInitialization=MI.SmallUniformWeights(), bInitialization=MI.ZeroBias(), epsilon=1e-6) :
		Decorator_ABC.__init__(self)
		self.epsilon = epsilon
		self.WInitialization = WInitialization
		self.bInitialization = bInitialization
		self.W = None
		self.b = None
		self.paramShape = None

	def getParameterShape(self, *args, **kwargs) :
		return self.paramShape

	def decorate(self, layer) :
		if not hasattr(layer, "batchnorm_W") or not hasattr(layer, "batchnorm_b") :
			self.paramShape = layer.getOutputShape()#(layer.nbOutputs, )
			self.WInitialization.initialize(self)
			self.bInitialization.initialize(self)

			layer.batchnorm_W = self.W
			layer.batchnorm_b = self.b

			mu = tt.mean(layer.outputs)
			sigma = tt.sqrt( tt.var(layer.outputs) + self.epsilon )
			layer.outputs = layer.batchnorm_W * ( (layer.outputs - mu) / sigma ) + layer.batchnorm_b

			mu = tt.mean(layer.testOutputs)
			sigma = tt.sqrt( tt.var(layer.testOutputs) + self.epsilon )
			layer.testOutputs = layer.batchnorm_W * ( (layer.testOutputs - mu) / sigma ) + layer.batchnorm_b
	
class WeightSparsity(Decorator_ABC):
	"""Stochatically sets a certain ratio of the input weight to 0"""
	def __init__(self, ratio, *args, **kwargs):
		Decorator_ABC.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1)
		self.ratio = ratio
		self.seed = MSET.RANDOM_SEED
		self.hyperParameters.extend(["ratio"])

	def decorate(self, layer) :
		initWeights = layer.W.get_value()
		for i in xrange(initWeights.shape[0]) :
			for j in xrange(initWeights.shape[1]) :
				if numpy.random.rand() < self.ratio :
					initWeights[i, j] = 0
	
		layer.W = theano.shared(value = initWeights, name = layer.W.name)

class InputSparsity(Decorator_ABC):
	"""Stochastically sets a certain ratio of the input connections to 0"""
	def __init__(self, ratio, *args, **kwargs):
		Decorator_ABC.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1)
		self.ratio = ratio
		self.seed = MSET.RANDOM_SEED
		self.hyperParameters.extend(["ratio"])

	def decorate(self, layer) :
		initWeights = layer.W.get_value()
		for i in xrange(initWeights.shape[0]) :
			if numpy.random.rand() < self.ratio :
				initWeights[i, : ] = numpy.zeros(initWeights.shape[1])
	
		layer.W = theano.shared(value = initWeights, name = layer.W.name)

	
