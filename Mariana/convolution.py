import theano, numpy, time
import theano.tensor as tt

import Mariana.layers as ML


class ConvPooler(object) :

	def __init__(self, *args, **kwrags) :
		pass
		
	def pool(self, convLayer) :
		raise NotImplemented("Must be implemented in child")

class Pass(ConvPooler) :
	"""No pooling"""
	def pool(self, convLayer) :
		hOutputs = convLayer.imageHeight - convLayer.filterHeight + 1
		wOutputs = convLayer.imageWidth - convLayer.filterWidth + 1
		nbOutputs = convLayer.nbMaps * hOutputs *  wOutputs
		
		return convLayer.convolution, nbOutputs

class MaxPooling2D(ConvPooler) :

	def __init__(self, heightDownscaleFactor, widthDownscaleFactor, *args, **kwargs) :
		"""downScaleFactors is the factor by which to downscale vertical and horizontal dimentions. (2,2) will halve the image in each dimension."""
		ConvPooler.__init__(self, *args, **kwargs)
		self.heightDownscaleFactor = heightDownscaleFactor
		self.widthDownscaleFactor = widthDownscaleFactor
	
	def pool(self, convLayer) :
		from theano.tensor.signal import downsample
		ds = (self.heightDownscaleFactor, self.widthDownscaleFactor)
		output = downsample.max_pool_2d( convLayer.convolution, ds, ignore_border = True )
		
		hOutputs = convLayer.imageHeight - convLayer.filterHeight + 1
		wOutputs = convLayer.imageWidth - convLayer.filterWidth + 1

		hRatio = hOutputs // self.heightDownscaleFactor
		wRatio = wOutputs // self.widthDownscaleFactor
		
		nbOutputs = convLayer.nbMaps * hRatio * wRatio
		
		return output, nbOutputs

class Flatten(ML.Layer_ABC) :

	def __init__(self, **kwargs) :
		"""Flattens the output of a convolution to a given numer of dimentions"""
		ML.Layer_ABC.__init__(self, None, **kwargs)
		self.outdim = 2

	def getParams(self) :
		return []

	def _setOutputs(self) :
		convLayer = self.feededBy.values()
		if len(convLayer) != 1 :
			raise ValueError("Flatten layers must have a single input")
		convLayer = convLayer[0]

		self.nbOutputs = convLayer.nbOutputs
		self.outputs = self.inputs.flatten(self.outdim)

class ConvolutionInput(ML.Layer_ABC) :
	def __init__(self, nbChannels, imageHeight, imageWidth, **kwargs) :
		ML.Layer_ABC.__init__(self, nbChannels, **kwargs)
		self.imageHeight = imageHeight
		self.imageWidth = imageWidth
		self.inputs = tt.matrix(name = self.name)
	
	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = self.inputs
		self._decorate()

class Convolution2D(ML.Layer_ABC) :

	def __init__(self, nbMaps, imageHeight, imageWidth, filterHeight, filterWidth, activation, pooler, **kwargs) :
		ML.Layer_ABC.__init__(self, nbMaps, **kwargs)
		self.nbMaps = nbMaps
		self.activation = activation
		self.imageHeight = imageHeight
		self.imageWidth = imageWidth
		self.filterHeight = filterHeight
		self.filterWidth = filterWidth
		self.pooler = pooler
		
	def _setOutputs(self) :
		from theano.tensor.nnet import conv

		inputLayer = self.feededBy.values()
		if len(inputLayer) != 1 :
			raise ValueError("Convolution layers must have a single input")
		inputLayer = inputLayer[0]

		if inputLayer.__class__ is Input :
			self.imageShape = (-1, 1, self.imageHeight, self.imageWidth)
			self.filterShape = (self.nbOutputs, 1, self.filterHeight, self.filterWidth) 
			self.inputs = self.inputs.reshape(self.imageShape)
		else :
			try :
				self.imageShape = (-1, inputLayer.nbMaps, self.imageHeight, self.imageWidth)
				self.filterShape = (self.nbOutputs, inputLayer.nbMaps, self.filterHeight, self.filterWidth) 
			except AttributeError :
				raise ValueError("Input to a conv2d layer must be of type Input or Convolution2D")

		initWeights = numpy.random.normal(0, 0.01, self.filterShape)
		initWeights = numpy.asarray(initWeights, dtype = theano.config.floatX)
		self.W = theano.shared(value = initWeights, name = self.name + "_W", borrow = True)
		
		initB = numpy.zeros((self.filterShape[0],), dtype = theano.config.floatX)
		self.b = theano.shared(value = initB, borrow = True)

		self.convolution = conv.conv2d( input = self.inputs, filters = self.W, filter_shape = self.filterShape )
		self.pooled, self.nbOutputs = self.pooler.pool(self)
		
		self.outputs = self.activation.function(self.pooled + self.b.dimshuffle('x', 0, 'x', 'x'))
		
	def getParams(self) :
		"""returns the layer parameters (Weights and bias)"""
		return [self.W, self.b]