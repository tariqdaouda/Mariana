from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

from types import *
import cPickle

import theano, numpy, time
import theano.tensor as tt
#import Mariana.rules as MR
import Mariana.activations as MA
from network import Network
from wrappers import TheanoFunction

class Layer_ABC(object) :
	"the abstract layer class"

	__metaclass__ = ABCMeta

	def __init__(self, nbOutputs, decorators = [], name = None) :
		
		if name is not None :
			self.name = name
		else :
			self.name = "%s_%s" %(self.__class__.__name__, numpy.random.random())

		self.type = "no-type-defined"

		self.nbInputs = None
		self.inputs = None
		self.nbOutputs = nbOutputs
		self.outputs = None
		self.params = []

		self.decorators = decorators

		self.feedsInto = OrderedDict()
		self.feededBy = OrderedDict()

		self.network = None
		self._inputRegistrations = set()
		
		self._mustInit = True

	def addDecorator(self, decorator) :
		self.decorators.append(decorator)

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		res = self.__class__(self.nbOutputs)
		res.W = self.W
		res.b = self.b
		res.name = self.name + "_clone"
		
		for k, v in kwargs.iteritems() :
			setattr(res, k, v)

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
				l._decorate()
			self._mustInit = False
	
	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		"""Sets the output of the layer. This function is called by _init() ans should be initialised in child"""
		raise NotImplemented("Should be implemented in child")
	
	def _decorate(self) :
		"""applies decorators"""
		for d in self.decorators :
			d.decorate(self)

	def _setNbInputs(self, layer) :
		"""Sets the size of input that the layer receives"""
		if self.nbInputs is not None :
			raise ValueError("A computation layer can only have one single input")
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

			if isinstance(layer, Output_ABC) or issubclass(layer.__class__, Output_ABC) :
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
		return '[label="%s: %sx%s"]' % (self.name, self.nbInputs, self.nbOutputs)

	def __len__(self) :
		return self.nbOutputs

class Input(Layer_ABC) :
	"An input layer"
	def __init__(self, nbInputs, name = None, **kwargs) :
		Layer_ABC.__init__(self, nbInputs, name = name, **kwargs)
		self.type = "input"
		self.nbInputs = nbInputs
		self.network = Network()#y, self)
		self.network.addInput(self)

	def _setOutputs(self) :
		"initialises the ouput to be the same as the inputs"
		self.outputs = tt.matrix(name = self.name + "_X")

	def _dot_representation(self) :
		return '[label="%s: %s" shape=invtriangle]' % (self.name, self.nbOutputs)

class Composite(Layer_ABC):
	"""A Composite layer is a layer that brings together the outputs of several other layers
	for example is we have::
		
		c = Composite()
		layer1 > c
		layer2 > c

	The output of c will be single vector: [layer1.output, layer2.output]
	"""
	def __init__(self, name = None):
		Layer_ABC.__init__(self, nbOutputs = None, name = name)

	def _setNbInputs(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = 0
		self.nbInputs += layer.nbOutputs
	
	def _setOutputs(self) :
		self.nbOutputs = self.nbInputs
		self.outputs = tt.concatenate( [l.outputs for l in self.feededBy.itervalues()], axis = 1 )

	def _dot_representation(self) :
		return '[label="%s: %s" shape=box]' % (self.name, self.nbOutputs)
		
class Hidden(Layer_ABC) :
	"A basic hidden layer"
	def __init__(self, nbOutputs, activation = getattr(MA, "tanh"), learningScenario = None, name = None, regularizations = [], **kwargs) :
		Layer_ABC.__init__(self, nbOutputs, name = name, **kwargs)
		self.W = None
		self.b = None

		self.type = "hidden"
		self.activation = activation
		self.learningScenario = learningScenario
		
		self.regularizationObjects = regularizations
		self.regularizations = []

	def _setOutputs(self) :
		"""initialises weights and bias. By default weights are setup to random low values, use mariana decorators
		to change this behaviour."""

		if self.W is None :
			initWeights = numpy.random.random((self.nbInputs, self.nbOutputs))
			initWeights = initWeights/sum(initWeights)
			initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)

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
	
		for reg in self.regularizationObjects :
			self.regularizations.append(reg.getFormula(self))

	def toOutput(self, outputType, **kwargs) :
		"returns an output layer with the same activation function, weights and bias"
		o = outputType(self.nbOutputs, activation = self.activation, name = self.name + '_toOutput', **kwargs)
		o.W = self.W
		o.b = self.b
		return o

class Output_ABC(Hidden) :
	"""An output layer"""
	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		"""The output layer defines the learning rate (lr), as well as any other parameters related to the learning"""

		Hidden.__init__(self, nbOutputs, activation = activation, name = name, **kwargs)
		self.type = "output"
		self.target = None
		self.dependencies = OrderedDict()
		self.costObject = costObject
		self.learningScenario = learningScenario

	def _backTrckDependencies(self) :
		"""Finds all the hidden layers the ouput layer is influenced by"""
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

		self._backTrckDependencies()
		cost = self.costObject.getCost(self)
	
		for l in self.dependencies.itervalues() :
			if l.__class__  is not Composite :
				for reg in l.regularizations :
					cost += reg

		updates = self.learningScenario.getUpdates(self, cost)

		for l in self.dependencies.itervalues() :
			try :
				updates.extend(l.learningScenario.getUpdates(l, cost))
			except AttributeError :
				updates.extend(self.learningScenario.getUpdates(l, cost))

		self.train = TheanoFunction("train", self, [cost, self.outputs], { "target" : self.target }, updates = updates)
		self.test = TheanoFunction("test", self, [cost, self.outputs], { "target" : self.target })
		self.propagate = TheanoFunction("propagate", self, [self.outputs])
	
		self._setTheanoFunctions()

	def _setTheanoFunctions(self) :
		"This is where you should put the definitions of your custom theano functions"
		pass

	def toHidden(self, **kwargs) :
		"returns a hidden layer with the same activation function, weights and bias"
		h = Hidden(self.nbOutputs, activation = self.activation, name = self.name + "_toHidden", **kwargs)
		h.W = self.W
		h.b = self.b
		return h

	def _dot_representation(self) :
		return '[label="%s: %sx%s" shape=invtriangle]' % (self.name, self.nbInputs, self.nbOutputs)

class Classifier_ABC(Output_ABC):
		"An abstract Classifier"
		def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
			Output_ABC.__init__(self, nbOutputs, activation, learningScenario, costObject, name, **kwargs)

		def _setTheanoFunctions(self) :
			"""Classifiers must define a 'classify' function that returns the result of the classification"""
			raise NotImplemented("theano classify must be defined in child's _setTheanoFunctions()")				

class SoftmaxClassifier(Classifier_ABC) :
	"""A softmax Classifier"""
	def __init__(self, nbOutputs, learningScenario, costObject, name = None, **kwargs) :
		Classifier_ABC.__init__(self, nbOutputs, activation = MA.softmax, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.target = tt.ivector(name = self.name + "_Target")

	def _setTheanoFunctions(self) :
		"""defined theano_classify, that returns the argmax of the output"""
		self.classify = TheanoFunction("classify", self, [ tt.argmax(self.outputs) ])

class Regression(Output_ABC) :
	"""For regressions"""
	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		Output_ABC.__init__(self, nbOutputs, activation = activation, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.target = tt.matrix(name = self.name + "_Target")