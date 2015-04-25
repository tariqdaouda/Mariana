import numpy, time
import theano
import theano.tensor as tt

class Decorator(object) :
	"""A decorator is modifier that is applied on the layer. Use them to define special kind of initialisations for the weight,
	or transformations that you want to apply on your layers durig training"""
	def __init__(self, *args, **kwargs) :
		self.hyperParameters = []
		self.name = self.__class__.__name__

	def __call__(self, *args, **kwargs) :
		self.decorate(*args, **kwargs)

	def decorate(self, layer) :
		"""The function that all decorators must implement"""
		raise NotImplemented("This one should be implemented in child")

class GlorotTanhInit(Decorator) :
	"""Set up the layer to apply the tanh initialisation suggested by Glorot et al."""
	def __init__(self, *args, **kwargs) :
		Decorator.__init__(self, *args, **kwargs)

	def decorate(self, layer) :
		rng = numpy.random.RandomState(int(time.time()))
		initWeights = rng.uniform(
					low = -numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
					high = numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
					size = (layer.nbInputs, layer.nbOutputs)
				)
		initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		layer.W = theano.shared(value = initWeights, name = layer.name + "_Glorot_tanh_W")

class StochasticTurnOff(Decorator):
	"""Applies StochasticTurnOff with a given ratio to a layer. Use it to make things such as denoising autoencoders and dropout layers"""
	def __init__(self, ratio, *args, **kwargs):
		# super(StochasticTurnOff, self).__init__()
		Decorator.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1) 
		self.ratio = ratio
		self.seed = time.time()
		self.hyperParameters = ["ratio"]

	def decorate(self, layer) :
		rnd = tt.shared_randomstreams.RandomStreams()
		mask = rnd.binomial(n = 1, p = (1-self.ratio), size = layer.outputs.shape)
		#cast to stay in GPU float limit
		mask = tt.cast(mask, theano.config.floatX)
		layer.outputs = layer.outputs * mask
		layer.name += "_drop_%s" % self.ratio

class WeightSparsity(Decorator):
	"""Stochatically sets a certain ratio of the weight to 0"""
	def __init__(self, ratio, *args, **kwargs):
		# super(StochasticTurnOff, self).__init__()
		Decorator.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1) 
		self.ratio = ratio
		self.seed = time.time()
		self.hyperParameters = ["ratio"]

	def decorate(self, layer) :
		initWeights = layer.W.get_value()
		for i in xrange(initWeights.shape[0]) :
			for j in xrange(initWeights.shape[1]) :
				if numpy.random.rand() < self.ratio :
					initWeights[i, j] = 0
		
		layer.name += "_ws_%s" % self.ratio
		layer.W = theano.shared(value = initWeights, name = layer.name)

class InputSparsity(Decorator):
	"""Stochatically sets a certain ratio of the input connections to 0"""
	def __init__(self, ratio, *args, **kwargs):
		# super(StochasticTurnOff, self).__init__()
		Decorator.__init__(self, *args, **kwargs)

		assert (ratio >= 0 and ratio <= 1) 
		self.ratio = ratio
		self.seed = time.time()
		self.hyperParameters = ["ratio"]

	def decorate(self, layer) :
		initWeights = layer.W.get_value()
		for i in xrange(initWeights.shape[0]) :
			if numpy.random.rand() < self.ratio :
				initWeights[i, : ] = numpy.zeros(initWeights.shape[1])
		
		layer.name += "_is_%s" % self.ratio
		layer.W = theano.shared(value = initWeights, name = layer.name)

		