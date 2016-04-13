import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
from Mariana.abstraction import Abstraction_ABC

__all__= ["Decorator_ABC", "OutputDecorator_ABC", "BinomialDropout", "WeightSparsity", "InputSparsity"]

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

class BinomialDropout(OutputDecorator_ABC):
	"""Use it to make things such as denoising autoencoders and dropout layers"""
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
		layer.outputs = self._decorate(layer.outputs)
		if not self.trainOnly :
			layer.testOutputs = self._decorate(layer.testOutputs)

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

		