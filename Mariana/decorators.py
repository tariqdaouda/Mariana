import numpy, time
import theano
import theano.tensor as tt
import Mariana.settings as MSET

__all__= ["Decorator_ABC", "OutputDecorator_ABC", "BinomialDropout", "GlorotTanhInit", "ZerosInit", "WeightSparsity", "InputSparsity"]

class Decorator_ABC(object) :
	"""A decorator is a modifier that is applied on a layer. This class defines the interface that a decorator
	must offer."""

	def __init__(self, *args, **kwargs) :
		self.hyperParameters = []
		self.name = self.__class__.__name__

	def __call__(self, *args, **kwargs) :
		self.decorate(*args, **kwargs)

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
		OutputDecorator_ABC.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1) 
		self.ratio = ratio
		self.seed = time.time()
		self.hyperParameters.extend(["ratio"])

	def decorate(self, layer) :
		rnd = tt.shared_randomstreams.RandomStreams()
		mask = rnd.binomial(n = 1, p = (1-self.ratio), size = layer.outputs.shape)
		#cast to stay in GPU float limit
		# mask = tt.cast(mask, theano.config.floatX)
		layer.outputs = layer.outputs * mask
		if not trainOnly :
			layer.test_outputs = layer.test_outputs * mask

class GlorotTanhInit(Decorator_ABC) :
	"""Set up the layer to apply the tanh initialisation introduced by Glorot et al. 2010"""
	def __init__(self, *args, **kwargs) :
		Decorator_ABC.__init__(self, *args, **kwargs)

	def decorate(self, layer) :
		rng = numpy.random.RandomState(MSET.RANDOM_SEED)
		initWeights = rng.uniform(
					low = -numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
					high = numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
					size = (layer.nbInputs, layer.nbOutputs)
				)

		initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		layer.W = theano.shared(value = initWeights, name = layer.W.name)

class ZerosInit(Decorator_ABC) :
	"""Initiales the weights at zeros"""
	def __init__(self, *args, **kwargs) :
		Decorator_ABC.__init__(self, *args, **kwargs)

	def decorate(self, layer) :
		initWeights = numpy.zeros(
					(layer.nbInputs, layer.nbOutputs),
					dtype = theano.config.floatX
				)

		initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		layer.W = theano.shared(value = initWeights, name = layer.W.name)
	
class WeightSparsity(Decorator_ABC):
	"""Stochatically sets a certain ratio of the input weight to 0"""
	def __init__(self, ratio, *args, **kwargs):
		Decorator_ABC.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1) 
		self.ratio = ratio
		self.seed = time.time()
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
		self.seed = time.time()
		self.hyperParameters.extend(["ratio"])

	def decorate(self, layer) :
		initWeights = layer.W.get_value()
		for i in xrange(initWeights.shape[0]) :
			if numpy.random.rand() < self.ratio :
				initWeights[i, : ] = numpy.zeros(initWeights.shape[1])
		
		layer.W = theano.shared(value = initWeights, name = layer.W.name)

		