import theano, numpy, time
import theano.tensor as tt

import Mariana.layers as ML
import Mariana.initializations as MI
import Mariana.network as MNET

__all__ = ["ConvPooler_ABC", "NoPooling", "MaxPooling2D", "Flatten", "ConvLayer_ABC", "InputChanneler", "Input", "Convolution2D"]

class ConvPooler_ABC(object) :
	"""The interface that all poolers must implement"""
	def __init__(self, *args, **kwrags) :
		self.hyperParameters = []
	
	def apply(self, layer) :
		hyps = {}
		for k in self.hyperParameters :
			hyps[k] = getattr(self, k)

		message = "%s uses pooler %s" % (layer.name, self.__class__.__name__)
		layer.network.logLayerEvent(layer, message, hyps)
		return self.pool(layer)

	def getOutputHeight(self, convLayer) :
		"""Returns image height after pooling"""
		raise NotImplemented("Must be implemented in child")

	def getOutputWidth(self, convLayer) :
		"""Returns image width after pooling"""
		raise NotImplemented("Must be implemented in child")
	
	def pool(self, convLayer) :
		"""This function takes the convolution layer and performs some pooling (usually down sampling).
		It must return a tuple (outputs, mapHeight, mapWidth)
		"""
		raise NotImplemented("Must be implemented in child")

class NoPooling(ConvPooler_ABC) :
	"""No pooling. The convolution is kept as is"""
	
	def getOutputHeight(self, convLayer) :
		"""Returns image height after pooling"""
		hOutputs = convLayer.inputHeight - convLayer.filterHeight + 1
		return hOutputs

	def getOutputWidth(self, convLayer) :
		"""Returns image width after pooling"""
		wOutputs = convLayer.inputWidth - convLayer.filterWidth + 1
		return wOutputs

	def pool(self, convLayer) :
		return convLayer.convolution

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
		self.hyperParameters = ['heightDownscaleFactor', 'widthDownscaleFactor']
	
	def getOutputHeight(self, convLayer) :
		hOutputs = convLayer.inputHeight - convLayer.filterHeight + 1
		hImage = hOutputs // self.heightDownscaleFactor
		return hImage
		
	def getOutputWidth(self, convLayer) :
		wOutputs = convLayer.inputWidth - convLayer.filterWidth + 1
		wImage = wOutputs // self.widthDownscaleFactor
		return wImage

	def pool(self, convLayer) :
		from theano.tensor.signal import pool
		ds = (self.heightDownscaleFactor, self.widthDownscaleFactor)
		output = pool.pool_2d( convLayer.convolution, ds, ignore_border = True )
		
		return output

class Flatten(ML.Layer_ABC) :

	def __init__(self, **kwargs) :
		"""Flattens the output of a convolution layer so it can be fed into a regular layer"""
		ML.Layer_ABC.__init__(self, None, layerType=MNET.TYPE_HIDDEN_LAYER, **kwargs)
		self.outdim = 2
		
		self.inputHeight = None
		self.inputWidth = None
		self.nbInputChannels = None

	def _femaleConnect(self, layer) :
		if self.nbOutputs is None :
			self.nbOutputs = layer.nbFlatOutputs
		elif self.nbOutputs != layer.nbFlatOutputs :
			raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, layer.nbFlatOutputs) )
	
	def _setOutputs(self) :
		for inputLayer in self.network.inConnections[self] :
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

		self.nbInputs = self.nbInputChannels
		self.outputs = self.inputs.flatten(self.outdim)
		self.testOutputs = self.inputs.flatten(self.outdim)
	
	def _dot_representation(self) :
		return '[label="%s: %s->%s" shape=invhouse]' % (self.name, self.nbInputChannels, self.nbOutputs)

class ConvLayer_ABC(object) :
	"""The abstract class that all convolution layers must implement"""

	def __init__(self, nbChannels, **kwargs) :
		self.nbChannels = nbChannels
		self.height = None
		self.width = None
		self.nbFlatOutputs = None #the number of outputs flattened in 2d.

	def _dot_representation(self) :
		return '[label="%s: %sx%sx%s" shape=invhouse]' % ( self.name, self.nbChannels, self.height, self.width)

class InputChanneler(ConvLayer_ABC, ML.Layer_ABC) :
	"""Takes the outputs of several regular layer and stacks them into separate channels so they can be passed to a conv layer (All inputs must have the same dimentions).
	Channelers can also be very useful even if you have a single input that is not a 2D array. They will reshape your data
	to make in iot fit into a conv layer much faster than numpy."""
	def __init__(self, height, width, **kwargs) :
		"""
		:param int height: Image height.
		:param int width: Image width.
		"""
		ConvLayer_ABC.__init__(self, None, **kwargs)
		ML.Layer_ABC.__init__(self, None, layerType=MNET.TYPE_HIDDEN_LAYER, **kwargs)
		self.height = height
		self.width = width

	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = layer.nbOutputs
		elif self.nbInputs != layer.nbOutputs :
			raise ValueError("Input size to '%s' has previously been set to: %s. But '%s' has %s outputs" % (self.name, self.nbInputs, layer.name, layer.nbOutputs)) 
		
		if not self.nbChannels :
			self.nbChannels = 1
		else :
			self.nbChannels += 1
		
		if not self.nbOutputs :
			self.nbOutputs = 1
		else :
			self.nbOutputs += 1

		self.nbFlatOutputs = self.nbChannels * self.height * self.width
	
	def _setOutputs(self) :
		inps = []
		for l in self.network.inConnections[self] :
			inps.append(l.outputs)

		self.outputs = tt.stack(inps).reshape((-1, self.nbChannels, self.height, self.width))
		self.testOutputs = tt.stack(inps).reshape((-1, self.nbChannels, self.height, self.width))

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
		ML.Layer_ABC.__init__(self, nbChannels, layerType=MNET.TYPE_INPUT_LAYER, **kwargs)

		self.height = height
		self.width = width
		self.nbInputs = nbChannels
		
		self.inputs = tt.tensor4(name = self.name)
		self.nbFlatOutputs = self.height * self.width * self.nbChannels

	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = self.inputs
		self.testOutputs = self.inputs

class Embedding(ConvLayer_ABC, ML.Embedding) :
	"""This input layer will take care of creating the embeddings and training them. Embeddings are learned representations
	of the inputs that are much loved in NLP."""

	def __init__(self, size, nbDimentions, dictSize, initializations = [MI.SmallUniformEmbeddings()], **kwargs) :
		"""
		:param size int: the size of the input vector (if your input is a sentence this should be the number of words in it).
		:param nbDimentions int: the number of dimentions in wich to encode each word.
		:param dictSize int: the total number of words. 
		"""
		ConvLayer_ABC.__init__(self, nbDimentions, **kwargs)
		ML.Embedding.__init__(self, size, nbDimentions, dictSize, initializations=initializations, **kwargs)
		
		self.nbInputs = size
		self.nbOutputs = self.nbDimentions*self.nbInputs
		
		self.height = 1
		self.width = size
		self.shape = (self.dictSize, 1, self.nbDimentions)

		self.embeddings = None

	def getParameterShape(self, param) :
		if param == "embeddings" :
			return (self.dictSize, self.nbDimentions, 1)
		else :
			raise ValueError("Unknow parameter: %s" % param)

	def _setOutputs(self) :
		self.preOutputs = self.embeddings[self.inputs]
		self.outputs = self.preOutputs.reshape((self.inputs.shape[0], self.nbChannels, self.height, self.width))
		self.testOutputs = self.preOutputs.reshape((self.inputs.shape[0], self.nbChannels, self.height, self.width))
		
	def _dot_representation(self) :
		return '[label="%s: %s" shape=invhouse]' % (self.name, self.shape)

class Convolution2D(ConvLayer_ABC, ML.Hidden) :
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

	def _femaleConnect(self, layer) :
		try :
			if self.nbInputChannels is None :
				self.nbInputChannels = layer.nbChannels
				self.nbInputs = self.nbInputChannels
			elif self.nbInputChannels != layer.nbChannels :
				raise ValueError("Number of input channels to '%s' as previously been set to: %s. But '%s' has %s channels" % (self.name, self.nbInputChannels, layer.name, layer.nbChannels))
			
			if self.inputHeight is None :
				self.inputHeight = layer.height
			elif self.inputHeight != layer.height :
				raise ValueError("Input height to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputHeight, layer.name, layer.height))
			
			if self.inputWidth is None :
				self.inputWidth = layer.width
			elif self.inputWidth != layer.width :
				raise ValueError("Input width to '%s' as previously been set to: %s. But '%s' is %s" % (self.name, self.inputWidth, layer.name, layer.width))
		
		except AttributeError :
			raise ValueError("Input must be a convolution layer")

		self.height = self.pooler.getOutputHeight(self)
		self.width = self.pooler.getOutputWidth(self)
		self.nbFlatOutputs = self.nbChannels * self.height * self.width
	
	def getParameterShape(self, param) :
		if param == "W" :
			return (self.nbOutputs, self.nbInputChannels, self.filterHeight, self.filterWidth) 
		elif param == "b" :
			return (self.nbOutputs,)
		else :
			raise ValueError("Unknow parameter: %s" % param)

	def _setOutputs(self) :
		from theano.tensor.nnet import conv
		
		for layer in self.network.inConnections[self] :
			
			if self.inputs is None :
				self.inputs = layer.outputs
			else :
				self.inputs += layer.outputs

		if self.filterHeight > self.inputHeight :
			raise ValueError("Filter height for '%s' cannot be bigger than its input height: '%s' > '%s'" % (self.name, self.filterHeight, self.inputHeight))

		if self.filterWidth > self.inputWidth :
			raise ValueError("Filter width for '%s' cannot be bigger than its input width: '%s' > '%s'" % (self.name, self.filterWidth, self.inputWidth))

		self.convolution = conv.conv2d( input = self.inputs, filters = self.W, filter_shape = self.getParameterShape('W') )
		self.pooled = self.pooler.apply(self)
		self.nbFlatOutputs = self.nbChannels * self.height * self.width

		if self.b is None:
			MI.ZerosBias().apply(self)
		
		self.b = self.b.dimshuffle('x', 0, 'x', 'x')

		self.outputs = self.pooled + self.b
		self.testOutputs = self.pooled + self.b