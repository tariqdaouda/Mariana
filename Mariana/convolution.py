import theano, numpy, time
import theano.tensor as tt

import Mariana.layers as ML
import Mariana.network as MNET

__all__ = ["ConvPooler_ABC", "NoPool", "MaxPooling2D", "Flatten", "ConvLayer_ABC", "Composite", "InputChanneler", "Input", "Convolution2D"]

class ConvPooler_ABC(object) :
	"""The interface that all poolers must implement"""
	def __init__(self, *args, **kwrags) :
		pass
		
	def pool(self, convLayer) :
		"""This function takes the convolution layer and performs some pooling (usually down sampling).
		It must return a tuple (outputs, mapHeight, mapWidth)
		"""
		raise NotImplemented("Must be implemented in child")

class NoPooling(ConvPooler_ABC) :
	"""No pooling. The convolution is kept as is"""
	def pool(self, convLayer) :
		hOutputs = convLayer.inputHeight - convLayer.filterHeight + 1
		wOutputs = convLayer.inputWidth - convLayer.filterWidth + 1
		
		return convLayer.convolution, hOutputs, wOutputs

class MaxPooling2D(ConvPooler_ABC) :
	"""
	Popular downsampling. This will divide each feature map into a number of independent smaller squares
	and take the activation of the most activated neurone in each square
	"""
	def __init__(self, heightDownscaleFactor, widthDownscaleFactor) :
		"""
		:param int heightDownscaleFactor: Factor by which to downscale the height of feature maps. Ex: 2 will halve the height.
		:param int widthDownscaleFactor: Factor by which to downscale the width of feature maps. Ex: 2 will halve the width.
		"""
		ConvPooler_ABC.__init__(self)
		self.heightDownscaleFactor = heightDownscaleFactor
		self.widthDownscaleFactor = widthDownscaleFactor
	
	def pool(self, convLayer) :
		from theano.tensor.signal import downsample
		ds = (self.heightDownscaleFactor, self.widthDownscaleFactor)
		output = downsample.max_pool_2d( convLayer.convolution, ds, ignore_border = True )
		
		hOutputs = convLayer.inputHeight - convLayer.filterHeight + 1
		wOutputs = convLayer.inputWidth - convLayer.filterWidth + 1

		hImage = hOutputs // self.heightDownscaleFactor
		wImage = wOutputs // self.widthDownscaleFactor
		
		return output, hImage, wImage

class Flatten(ML.Layer_ABC) :

	def __init__(self, **kwargs) :
		"""Flattens the output of a convolution layer so it can be fed into a regular layer"""
		ML.Layer_ABC.__init__(self, None, **kwargs)
		self.type = ML.TYPE_HIDDEN_LAYER
		self.outdim = 2
		
		self.inputHeight = None
		self.inputWidth = None
		self.nbInputChannels = None

	def getParams(self) :
		return []

	def _setOutputs(self) :
		for inputLayer in self.feededBy.itervalues() :
			if self.nbInputChannels is None :
				self.nbInputChannels = inputLayer.nbChannels
			elif self.nbInputChannels != inputLayer.nbChannels :
				raise ValueError("Number of input channels to '%s' as previously been set to: %s. But '%s' has %s channels" % (self.name, self.nbInputChannels, inputLayer.name, inputLayer.nbChannels))
			
			if self.inputHeight is None :
				self.inputHeight = inputLayer.height
			elif self.inputHeight != inputLayer.height :
				raise ValueError("Input height to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputHeight, inputLayer.name, inputLayer.height))
			
			if self.inputWidth is None :
				self.inputWidth = inputLayer.width
			elif self.inputWidth != inputLayer.width :
				raise ValueError("Input width to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputWidth, inputLayer.name, inputLayer.width))

			if self.inputs is None :
				self.inputs = inputLayer.outputs
				self.nbOutputs = inputLayer.nbFlatOutputs
			else :
				self.inputs += inputLayer.outputs

		self.nbInputs = self.nbInputChannels
		self.outputs = self.inputs.flatten(self.outdim)
		self._decorate()
	
	def _dot_representation(self) :
		return '[label="%s: %s->%s" shape=invhouse]' % (self.name, self.nbInputChannels, self.nbOutputs)

class ConvLayer_ABC(object) :
	"""The abstract class that all convolution layers must implement"""

	def __init__(self, nbChannels, **kwargs) :
		self.nbChannels = nbChannels
		self.height = None
		self.width = None
		self.nbFlatOutputs = None #the number of outputs flattened in 2d.

class InputChanneler(ConvLayer_ABC, ML.Layer_ABC) :
	"""Takes the outputs of several regular layer and pools them into separate channels. All inputs must have the same dimentions"""
	def __init__(self, height, width, **kwargs) :
		"""
		:param int height: Image height.
		:param int width: Image width.
		"""
		ConvLayer_ABC.__init__(self, None, **kwargs)
		ML.Layer_ABC.__init__(self, None, **kwargs)
		self.height = height
		self.width = width
	
	def getParams(self) :
		return []

	def _setOutputs(self) :
		inps = []
		for l in self.feededBy.itervalues() :
			if self.nbInputs is None :
				self.nbInputs = l.nbOutputs
			elif self.nbInputs != l.nbOutputs :
				raise ValueError("Input size to '%s' has previously been set to: %s. But '%s' has %s outputs" % (self.name, self.nbInputs, l.name, l.nbOutputs)) 
			inps.append(l.outputs)

		self.nbChannels = len(inps)
		self.nbOutputs = len(inps)
		self.outputs = tt.stack(inps).reshape((-1, self.nbChannels, self.height, self.width))
		self.nbFlatOutputs = self.nbChannels * self.height * self.width
		self._decorate()

class Input(ConvLayer_ABC, ML.Layer_ABC) :
	"""The input to a convolution network. This is different from a regular Input layer in the sense that it also holds channels information.
	To feed regular layers into a convolution network, have a look at InputChanneler."""

	def __init__(self, nbChannels, height, width, **kwargs) :
		"""
		:param int nbChannels: Number of channels in the images (ex: RGB is 3).
		:param int height: Image height.
		:param int width: Image width.
		"""
		ConvLayer_ABC.__init__(self, nbChannels, **kwargs)
		ML.Layer_ABC.__init__(self, nbChannels, **kwargs)

		self.type = ML.TYPE_INPUT_LAYER
		self.height = height
		self.width = width
		self.nbInputs = nbChannels
		self.network = MNET.Network()
		self.network.addInput(self)
		
		self.inputs = tt.tensor4(name = self.name)
		self.nbFlatOutputs = self.height * self.width * self.nbChannels

	def getParams(self) :
		return []

	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = self.inputs
		self.test_outputs = self.inputs
		self._decorate()

class Convolution2D(ML.Hidden, ConvLayer_ABC) :
	"""The layer that performs the convolutions"""

	def __init__(self, nbFilters, filterHeight, filterWidth, activation, pooler, **kwargs) :
		"""
		:param int nbFilters: Number of filters (feature maps) generated by the network.
		:param int filterHeight: Height of each filter.
		:param int filterWidth: Width of each filter.
		:param Activation activation: An activation object such as Tanh or ReLU.
		:param Pooler pooler: A pooler object.
		"""
		ConvLayer_ABC.__init__(self, nbFilters)
		ML.Hidden.__init__(self, nbFilters, activation = activation, **kwargs)

		self.inputHeight = None
		self.inputWidth = None
		self.filterHeight = filterHeight
		self.filterWidth = filterWidth
		self.nbInputChannels = None
		self.pooler = pooler
		self.nbFlatOutputs = None

	def _addInputs(self) :
		try :
			for inputLayer in self.feededBy.itervalues() :
				if self.nbInputChannels is None :
					self.nbInputChannels = inputLayer.nbChannels
				elif self.nbInputChannels != inputLayer.nbChannels :
					raise ValueError("Number of input channels to '%s' as previously been set to: %s. But '%s' has %s channels" % (self.name, self.nbInputChannels, inputLayer.name, inputLayer.nbChannels))
				
				if self.inputHeight is None :
					self.inputHeight = inputLayer.height
				elif self.inputHeight != inputLayer.height :
					raise ValueError("Input height to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputHeight, inputLayer.name, inputLayer.height))
				
				if self.inputWidth is None :
					self.inputWidth = inputLayer.width
				elif self.inputWidth != inputLayer.width :
					raise ValueError("Input width to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputWidth, inputLayer.name, inputLayer.width))

				if self.inputs is None :
					self.inputs = inputLayer.outputs
				else :
					self.inputs += inputLayer.outputs
		except AttributeError :
			raise ValueError("Input must be a convolution layer")

		if self.filterHeight > self.inputHeight :
			raise ValueError("Filter height for '%s' cannot be bigger than its input height: '%s' > '%s'" % (self.name, self.filterHeight, self.inputHeight))

		if self.filterWidth > self.inputWidth :
			raise ValueError("Filter width for '%s' cannot be bigger than its input width: '%s' > '%s'" % (self.name, self.filterWidth, self.inputWidth))

		self.nbInputs = self.nbInputChannels

	def _setOutputs(self) :
		from theano.tensor.nnet import conv

		self._addInputs()

		self.filterShape = (self.nbOutputs, self.nbInputChannels, self.filterHeight, self.filterWidth) 

		initWeights = numpy.random.normal(0, 0.01, self.filterShape)
		initWeights = numpy.asarray(initWeights, dtype = theano.config.floatX)
		self.W = theano.shared(value = initWeights, name = self.name + "_W", borrow = True)
		
		initB = numpy.zeros((self.filterShape[0],), dtype = theano.config.floatX)
		self.b = theano.shared(value = initB, borrow = True)

		self.convolution = conv.conv2d( input = self.inputs, filters = self.W, filter_shape = self.filterShape )
		self.pooled, self.height, self.width = self.pooler.pool(self)
		self.nbFlatOutputs = self.nbChannels * self.height * self.width

		self.outputs = self.activation.function(self.pooled + self.b.dimshuffle('x', 0, 'x', 'x'))
		self._decorate()
		
	def getParams(self) :
		"""returns the layer parameters (Weights and bias)"""
		return [self.W, self.b]