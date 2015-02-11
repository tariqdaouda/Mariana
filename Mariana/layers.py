from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

from types import *
import cPickle

import theano, numpy, time
import theano.tensor as tt
import Mariana.costs as MC
from Network import Network
from wrappers import TheanoFunction

class LayerABC(object) :
	"the abstract layer class"

	__metaclass__ = ABCMeta

	def __init__(self, nbOutputs, name = None) :
		
		if name is not None :
			self.name = name
		else :
			self.name = "%s_%s" %(self.__class__.__name__, numpy.random.random())

		self.nbInputs = None
		self.inputs = None
		self.nbOutputs = nbOutputs
		self.outputs = None
		self.params = []

		self.feedsInto = OrderedDict()
		self.feededBy = OrderedDict()

		self.network = None
		self._inputRegistrations = set()
	
		self.W = None
		self.b = None
		
		self._mustInit = True

	def clone(self) :
		"""Returns a free layer with the same weights and bias"""
		res = self.__class__(self.nbOutputs)
		res.W = self.W
		res.b = self.b
		return res

	def _registerInput(self, inputLayer) :
		"Registers an input to the layer. This function is meant to be called by inputLayer"
		self._inputRegistrations.add(inputLayer.name)

	def _init(self) :
		"Initialise the layer making it ready for training. This function is automatically called before train/test etc..."
		if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.feededBy) ) :
			self._setOutputs()
			if self.outputs == None :
				raise ValueError("Invalid network, layer '%s' has no defined outputs" % self.name)

			for l in self.feedsInto.itervalues() :
				if l.inputs is None :
					l.inputs = self.outputs
				else :
					l.inputs += self.outputs
				l._setNbInputs(self)
				l._registerInput(self)
				l._init()
			self._mustInit = False
	
	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		"""Sets the output of the layer. This function is called by _init() ans should be initialised in child"""
		raise NotImplemented("Should be implemented in child")

	def _setNbInputs(self, layer) :
		"""Sets the size of input that the layer receives"""
		if self.nbInputs is not None :
			raise ValueError("A layer can only have one single input")
		self.nbInputs = layer.nbOutputs

	def connect(self, layerOrList) :
		"""Connect the layer to list of layers or to a single layer"""
		def _connectLayer(me, layer) :
			me.feedsInto[layer.name] = layer
			layer.feededBy[me.name] = me
			me.network.addEdge(me, layer)
			if layer.network is not None :
				me.network.merge(me, layer)
			layer.network = me.network

		if self.network is None :
			self.network = Network(self)

		if type(layerOrList) is ListType :
			raise NotImplemented("List as argument is deprecated")
			# for p in layerOrList :
			# 	layer = p[1].entryLayer
			# 	_connectLayer(self, layer)
			# 	self.network.merge(self, p[1])
			# 	layer.network = self.network
		else :
			layer = layerOrList
			if isinstance(layer, Input) :
				raise ValueError("Nothing can be connected to an input layer")

			if isinstance(layer, Output) or issubclass(layer.__class__, Output) :
				self.network.addOutput(layer)
			_connectLayer(self, layer)
			
		return self.network

	def __gt__(self, pathOrLayer) :
		"""Alias to connect, make it possible to write things such as layer1 > layer2"""
		return self.connect(pathOrLayer)

	def __repr__(self) :
		return "(Mariana %s '%s': %sx%s )" % (self.__class__.__name__, self.name, self.nbInputs, self.nbOutputs)

	def _dot_representation(self) :
		"returns the representation of the node in the graph DOT format"
		return '%s [label="%s: %sx%s"]' % (self.name, self.name, self.nbInputs, self.nbOutputs)

class Input(LayerABC) :
	"An input layer"
	def __init__(self, nbInputs, name = None) :
		LayerABC.__init__(self, nbInputs, name = name)
		self.nbInputs = nbInputs
		self.network = Network()#y, self)
		self.network.addInput(self)

	def _setOutputs(self) :
		"initialises the ouput to be the same as the inputs"
		self.outputs = tt.matrix(name = self.name + "_X")

	def _dot_representation(self) :
		return '%s [label="%s: %s" shape=invtriangle]' % (self.name, self.name, self.nbOutputs)

class Composite(LayerABC):
	"""A Composite layer is a layer that brings together the ourputs of several other layers
	for example is we have::
		
		c = Composite()
		layer1 > c
		layer2 > c

	The output of c will be single vector: [layer1.output, layer2.output]
	"""
	def __init__(self, name = None):
		LayerABC.__init__(self, nbOutputs = None, name = name)

	def _setNbInputs(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = 0
		self.nbInputs += layer.nbOutputs
	
	def _setOutputs(self) :
		self.nbOutputs = self.nbInputs
		self.outputs = tt.concatenate( [l.outputs for l in self.feededBy.itervalues()], axis = 1 )

	def _dot_representation(self) :
		return '%s [label="%s: %s" shape=box]' % (self.name, self.name, self.nbOutputs)
		
class Hidden(LayerABC) :
	"A basic hidden layer"
	def __init__(self, nbOutputs, activation = tt.tanh, name = None, sparsity = 0) :
		LayerABC.__init__(self, nbOutputs, name = name)
		self.activation = activation
		self.sparsity = sparsity

	def _setOutputs(self) :
		"""initialises weights and bias, If the activation fct in tanh, weights will be initialised using Glorot[10],
		if it is something else wight will simply have small values."""
		if self.W is None :
			rng = numpy.random.RandomState(int(time.time()))
			if self.activation is tt.tanh :
				initWeights = rng.uniform(
					low = -numpy.sqrt(6. / (self.nbInputs + self.nbOutputs)),
					high = numpy.sqrt(6. / (self.nbInputs + self.nbOutputs)),
					size = (self.nbInputs, self.nbOutputs)
				)
			else :
				# print "---->", self.inputs, self, (self.nbInputs, self.nbOutputs)
				initWeights = numpy.random.random((self.nbInputs, self.nbOutputs))
				initWeights = initWeights/sum(initWeights)

			initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		
			for i in xrange(initWeights.shape[0]) :
				for j in xrange(initWeights.shape[1]) :
					if numpy.random.random() < self.sparsity :
						initWeights[i, j] = 0

			self.W = theano.shared(value = initWeights, name = self.name + "_W")
		else :
			if self.W.get_value().shape != (self.nbInputs, self.nbOutputs) :
				raise ValueError("weights have shape %s, but the layer has %s inputs and %s outputs" % (self.W.get_value().shape, self.nbInputs, self.nbOutputs))

		if self.b is None :
			initBias = numpy.zeros((self.nbOutputs,), dtype=theano.config.floatX)
			self.b = theano.shared(value = initBias, name = self.name + "_b")
		else :
			if self.b.get_value().shape[0] != self.nbOutputs :
				raise ValueError("bias has a length of %s, but there are %s outputs" % (self.b.get_value().shape[0], self.nbOutputs))

		self.params = [self.W, self.b]
		self.network.addParams(self.params)

		if self.activation is None :
			self.outputs = tt.dot(self.inputs, self.W) + self.b
		else :
			# self.outputs = theano.printing.Print('this is a very important value for %s' % self.name)(self.activation(tt.dot(self.inputs, self.W) + self.b))
			self.outputs = self.activation(tt.dot(self.inputs, self.W) + self.b)

class Output(Hidden) :
	"""An output layer"""
	def __init__(self, nbOutputs, activation, costFct, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
		"""The output layer defines the learning rate (lr), as well as any other parameters related to the learning"""

		Hidden.__init__(self, nbOutputs, activation = activation, name = name)
		self.costFct = costFct
		self.lr = lr
		self.l1 = l1
		self.l2 = l2
		self.momentum = momentum	
		self.y = tt.ivector(name = self.name + "_Y")
		self.dependencies = OrderedDict()

	def _backTrckDependencies(self) :
		"""Finds all the layers the ouput layer is connected to"""
		def _bckTrk(deps, layer) :		
			for l in layer.feededBy.itervalues() :
				if l.__class__ is not Input :
					deps[l.name] = l
					_bckTrk(deps, l)
			return deps

		self.dependencies = _bckTrk(self.dependencies, self)

	def _setOutputs(self) :
		"""Initialises the layer by creating the weights and and bias.
		Creates theano_train/theano_test/theano_propagate functions and calls _setTheanoFunctions to 
		create user custom theano functions."""
		Hidden._setOutputs(self)

		# self.outputs = sum([l.outputs for l in self.feededBy.itervalues()])
		self._backTrckDependencies()
		for l in self.dependencies.itervalues() :
			self.params.extend(l.params)

		# self.inputLayer = self.network.entryLayer
		
		# l1Fct = lambda w: abs(w).sum 
		# l2Fct = lambda w: (w**2).sum 
		s1 = 0
		for l in self.dependencies.itervalues() :
			if l.W is not None :
				s1 += abs(l.W).sum()
		L1 = self.l1 * ( abs(self.W).sum() + s1 )

		s2 = 0
		for l in self.dependencies.itervalues() :
			if l.W is not None :
				s2 += (l.W**2).sum()
		L2 = self.l2 * ( (self.W**2).sum() + s2 )
		self.cost = self.costFct(self.y, self.outputs) + L1 + L2

		self.updates = []
		for param in self.params :
			gparam = tt.grad(self.cost, param)
			momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
			self.updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
			self.updates.append((param, param - self.lr * momentum_param))
			# self.updates.append((param, param - self.lr * gparam))

		self.train = TheanoFunction(self, [self.cost, self.outputs], { "target" : self.y }, updates = self.updates)
		self.test = TheanoFunction(self, [self.cost, self.outputs], { "target" : self.y })
		self.propagate = TheanoFunction(self, [self.outputs])
	
		self._setTheanoFunctions()


	def _setTheanoFunctions(self) :
		"This is where you should put the definitions of your custom theano functions"
		pass

	def toHidden(self) :
		"returns a hidden layer with the same activation function, weights and bias"
		h = Hidden(self.nbOutputs, activation = self.activation, name = self.name, sparsity = 0)
		h.W = self.W
		h.b = self.b
		return h

	def _dot_representation(self) :
		return '%s [label="%s: %sx%s" shape=invtriangle]' % (self.name, self.name, self.nbInputs, self.nbOutputs)

class ClassifierABC(Output):
		"An abstract Classifier"
		def __init__(self, nbOutputs, activation, costFct, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
			Output.__init__(self, nbOutputs, activation,  costFct, lr, l1, l2, momentum , name)

		def _setTheanoFunctions(self) :
			"""Classifiers must define a 'classify' function that returns the result of the classification"""
			raise NotImplemented("theano classify must be defined in child's _setTheanoFunctions()")				

class SoftmaxClassifier(ClassifierABC) :
	"""A softmax Classifier"""
	def __init__(self, nbOutputs, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
		ClassifierABC.__init__(self, nbOutputs, activation = tt.nnet.softmax, costFct = MC.negativeLogLikelihood, lr = lr, l1 = l1, l2 = l2, momentum = momentum, name = name)

	def _setTheanoFunctions(self) :
		"""defined theano_classify, that returns the argmax of the output"""
		self.classify = TheanoFunction(self, [ tt.argmax(self.outputs) ])
	