from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

from types import *
import cPickle

import theano, numpy, time
import theano.tensor as tt
import Mariana.activations as MA
import Mariana.settings as MSET

import Mariana.network as MNET
import Mariana.wrappers as MWRAP

__all__ = ["Layer_ABC", "Output_ABC", "Classifier_ABC", "Input", "Hidden", "Composite", "SoftmaxClassifier", "Regression"]

TYPE_INPUT_LAYER = 0
TYPE_OUTPUT_LAYER = 1
TYPE_HIDDEN_LAYER = 1

class Layer_ABC(object) :
	"The interface that every layer should expose"

	__metaclass__ = ABCMeta

	def __init__(self, size, saveOutputs = MSET.SAVE_OUTPUTS_DEFAULT, decorators = [], name = None) :
		
		if name is not None :
			self.name = name
		else :
			self.name = "%s_%s" %(self.__class__.__name__, numpy.random.random())

		self.type = "no-type-defined"

		self.nbInputs = None
		self.inputs = None
		self.test_inputs = None
		self.nbOutputs = size
		self.outputs = None # this is a symbolic var
		self.test_outputs = None # this is a symbolic var

		if saveOutputs :
			initLO = numpy.zeros(self.nbOutputs, dtype=theano.config.floatX)
			self.last_outputs = theano.shared(value = numpy.matrix(initLO) ) # this will be a shared variable with the last values of outputs
		else :
			self.last_outputs = None

		self.decorators = decorators

		self.feedsInto = OrderedDict()
		self.feededBy = OrderedDict()

		self.network = None
		self._inputRegistrations = set()
		
		self._mustInit = True

	def addDecorator(self, decorator) :
		"""Add a decorator to the layer"""
		self.decorators.append(decorator)

	def getParams(self) :
		"""returns the layer parameters"""
		raise NotImplemented("Should be implemented in child")

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		raise NotImplemented("Should be implemented in child")

	def _registerInput(self, inputLayer) :
		"Registers an input to the layer. This function is meant to be called by inputLayer"
		self._inputRegistrations.add(inputLayer.name)

	def _init(self) :
		"Initialise the layer making it ready for training. This function is automatically called before train/test etc..."
		if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.feededBy) ) :
			self._setOutputs()
			if self.outputs is None :
				raise ValueError("Invalid layer '%s' has no defined outputs" % self.name)

			for l in self.feedsInto.itervalues() :
				if l.inputs is None :
					l.inputs = self.outputs
					l.test_inputs = self.test_outputs
				else :
					l.inputs += self.outputs
					l.test_inputs += self.test_outputs

				l._setNbInputs(self)
				l._registerInput(self)
				l._init()
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
		"""Connect the layer to another one"""
		def _connectLayer(me, layer) :
			me.feedsInto[layer.name] = layer
			layer.feededBy[me.name] = me
			me.network.addEdge(me, layer)
			if layer.network is not None :
				me.network.merge(me, layer)
			layer.network = me.network

		if self.network is None :
			self.network = MNET.Network(self)

		if type(layerOrList) is ListType :
			raise NotImplemented("List as argument is deprecated")
		else :
			layer = layerOrList
			if isinstance(layer, Input) :
				raise ValueError("Nothing can be connected to an input layer")

			if isinstance(layer, Output_ABC) or issubclass(layer.__class__, Output_ABC) :
				self.network.addOutput(layer)
			_connectLayer(self, layer)
			
		return self.network

	def disconnect(self, layer) :
		"""Severs a connection"""
		del(me.feedsInto[layer.name])
		del(layer.feededBy[me.name])
		me.network.removeEdge(me, layer)
		me.network._mustInit = True

	def getOutputs(self) :
		"""Return the last outputs of the layer"""
		try :
			return self.last_outputs.get_value()
		except AttributeError :
			raise AttributeError("Impossible to get the last ouputs of this layer, if you want them to be stored create with saveOutputs = True")

	def __gt__(self, pathOrLayer) :
		"""Alias to connect, make it possible to write things such as layer1 > layer2"""
		return self.connect(pathOrLayer)

	def __div__(self, pathOrLayer) :
		"""Alias to disconnect, make it possible to write things such as layer1 / layer2"""
		return self.disconnect(pathOrLayer)

	def __repr__(self) :
		return "(Mariana %s '%s': %sx%s )" % (self.__class__.__name__, self.name, self.nbInputs, self.nbOutputs)

	def _dot_representation(self) :
		"returns the representation of the node in the graph DOT format"
		return '[label="%s: %sx%s"]' % (self.name, self.nbInputs, self.nbOutputs)

	def __len__(self) :
		return self.nbOutputs

class Input(Layer_ABC) :
	"An input layer"
	def __init__(self, size, name = None, **kwargs) :
		Layer_ABC.__init__(self, size, name = name, **kwargs)
		self.kwargs = kwargs
		self.type = TYPE_INPUT_LAYER
		self.nbInputs = size
		self.network = MNET.Network()
		self.network.addInput(self)

	def getParams(self) :
		"""return nothing"""
		return []

	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = tt.matrix(name = self.name + "_X")
		self.test_outputs = tt.matrix(name = self.name + "_X")
		self._decorate()

	def _dot_representation(self) :
		return '[label="%s: %s" shape=invhouse]' % (self.name, self.nbOutputs)

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		res = self.__class__(self.nbInputs, name = self.name, **self.kwargs)
		return res

class Composite(Layer_ABC):
	"""A Composite layer concatenates the outputs of several other layers
	for example is we have::
		
		c = Composite()
		layer1 > c
		layer2 > c

	The output of c will be single vector: [layer1.output, layer2.output]
	"""
	def __init__(self, name = None):
		Layer_ABC.__init__(self, nbOutputs = None, name = name)

	def getParams(self) :
		return []

	def _setNbInputs(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = 0
		self.nbInputs += layer.nbOutputs
	
	def _setOutputs(self) :
		self.nbOutputs = self.nbInputs
		self.outputs = tt.concatenate( [l.outputs for l in self.feededBy.itervalues()], axis = 1 )

	def _dot_representation(self) :
		return '[label="%s: %s" shape=tripleoctogon]' % (self.name, self.nbOutputs)
		
class Hidden(Layer_ABC) :
	"A hidden layer"
	def __init__(self, size, activation = getattr(MA, "reLU"), learningScenario = None, name = None, regularizations = [], **kwargs) :
		Layer_ABC.__init__(self, size, name = name, **kwargs)
		self.W = None
		self.b = None
		
		self.type = TYPE_HIDDEN_LAYER
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

		if self.activation is None :
			self.outputs = tt.dot(self.inputs, self.W) + self.b
			self.test_outputs = tt.dot(self.test_inputs, self.W) + self.b
		else :
			# self.outputs = theano.printing.Print('this is a very important value for %s' % self.name)(self.activation(tt.dot(self.inputs, self.W) + self.b))
			self.outputs = self.activation(tt.dot(self.inputs, self.W) + self.b)
			self.test_outputs = self.activation(tt.dot(self.test_inputs, self.W) + self.b)
		
		self._decorate()
	
		for reg in self.regularizationObjects :
			self.regularizations.append(reg.getFormula(self))

	def getW(self) :
		"""Return the weight values"""
		return self.W.get_value()

	def getParams(self) :
		"""returns the layer parameters (Weights and bias)"""
		return [self.W, self.b]

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		res = self.__class__(self.nbOutputs)
		res.W = self.W
		res.b = self.b
		res.name = self.name
		
		for k, v in kwargs.iteritems() :
			setattr(res, k, v)

		return res

	def toOutput(self, outputType, **kwargs) :
		"returns an output layer with the same activation function, weights and bias"
		if "activation" in kwargs :
			act = kwargs["activation"]
		else :
			act = self.activation

		o = outputType(self.nbOutputs, activation = act, name = self.name, **kwargs)
		o.W = self.W
		o.b = self.b
		return o

class Output_ABC(Hidden) :
	"""The interface that every output layer should expose. This interface also provides tyhe model functions:

		* train: upadates the parameters and returns the cost
		* test: returns the cost, ignores trainOnly decoartors
		* propagate: returns the outputs of the network, ignores trainOnly decoartors
		"""

	def __init__(self, size, activation, learningScenario, costObject, name = None, **kwargs) :
		
		Hidden.__init__(self, size, activation = activation, name = name, **kwargs)
		self.type = TYPE_OUTPUT_LAYER
		self.targets = None
		self.dependencies = OrderedDict()
		self.costObject = costObject
		self.learningScenario = learningScenario
		self.lastOutsTestUpdates = None
	
	def _backTrckDependencies(self) :
		"""Finds all the hidden layers the ouput layer is influenced by"""
		def _bckTrk(deps, layer) :		
			for l in layer.feededBy.itervalues() :
				if l.__class__ is not Input :
					deps[l.name] = l
					_bckTrk(deps, l)
			return deps

		self.dependencies = _bckTrk(self.dependencies, self)

	def _setTheanoFunctions(self) :
		"""Creates theano_train/theano_test/theano_propagate functions and calls setCustomTheanoFunctions to 
		create user custom theano functions."""

		self._backTrckDependencies()
		
		cost = self.costObject.costFct(self.targets, self.outputs)
		test_cost = self.costObject.costFct(self.targets, self.test_outputs)
		
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
			
		self.lastOutsTestUpdates = []
		for l in self.network.layers.itervalues() :
			if ( l.last_outputs is not None ) and ( l.outputs is not None ) :
				self.lastOutsTestUpdates.append( (l.last_outputs, l.test_outputs ) )
				updates.append( (l.last_outputs, l.outputs ) )


		self.train = MWRAP.TheanoFunction("train", MWRAP.TYPE_TRAIN, self, [cost], { "target" : self.targets }, updates = updates, allow_input_downcast=True)
		self.test = MWRAP.TheanoFunction("test", MWRAP.TYPE_TEST, self, [test_cost], { "target" : self.targets }, updates = self.lastOutsTestUpdates, allow_input_downcast=True)
		self.propagate = MWRAP.TheanoFunction("propagate", MWRAP.TYPE_TEST, self, [self.test_outputs], updates = self.lastOutsTestUpdates, allow_input_downcast=True)
	
		self.setCustomTheanoFunctions()

	def setCustomTheanoFunctions(self) :
		"""This is where you should put the definitions of your custom theano functions. Theano functions 
		must be declared as self attributes using a wrappers.TheanoFunction object, cf. wrappers documentation.
		"""
		pass

	def toHidden(self, **kwargs) :
		"returns a hidden layer with the same activation function, weights and bias"
		h = Hidden(self.nbOutputs, activation = self.activation, name = self.name + "_toHidden", **kwargs)
		h.W = self.W
		h.b = self.b
		return h

	def _dot_representation(self) :
		return '[label="%s: %sx%s" shape=invhouse]' % (self.name, self.nbInputs, self.nbOutputs)

class Classifier_ABC(Output_ABC):
	"""The interface that every classifier should expose. Classifiers should provide a model function
	classify, that returns the result of the classification, updates self.last_outputs and ignores trainOnly
	decorators"""

	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		Output_ABC.__init__(self, nbOutputs, activation, learningScenario, costObject, name, **kwargs)

	def setCustomTheanoFunctions(self) :
		"""Classifiers must define a 'classify' function that returns the result of the classification"""
		raise NotImplemented("theano classify must be defined in child's setCustomTheanoFunctions()")				

class SoftmaxClassifier(Classifier_ABC) :
	"""A softmax (probabilistic) Classifier"""
	def __init__(self, nbOutputs, learningScenario, costObject, name = None, **kwargs) :
		Classifier_ABC.__init__(self, nbOutputs, activation = MA.softmax, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.targets = tt.ivector(name = self.name + "_Target")

	def setCustomTheanoFunctions(self) :
		"""defined theano_classify, that returns the argmax of the output"""
		self.classify = MWRAP.TheanoFunction("classify", MWRAP.TYPE_TEST, self, [ tt.argmax(self.test_outputs) ], updates = self.lastOutsTestUpdates)

	def _dot_representation(self) :
		return '[label="%s: %s" shape=doublecircle]' % (self.name, self.nbOutputs)

class Regression(Output_ABC) :
	"""For regressions, works great with a mean squared error cost"""
	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		Output_ABC.__init__(self, nbOutputs, activation = activation, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.targets = tt.matrix(name = self.name + "_Target")

	def _dot_representation(self) :
		return '[label="%s: %s" shape=egg]' % (self.name, self.nbOutputs)

#work in progress
class Convolution2D(Hidden) :

	def __init__(self, nbMaps, height, width, *theanoArgs, **theanoKwArgs) :
		self.nbMaps = 32
		self.height = 3
		self.width = 3
		self.border_mode = border_mode
		self.theanoArgs = theanoArgs
		self.theanoKwArgs = self.theanoKwArgs

	def _setOutputs(self) :
		self.outputs = self.activation(conv2d(self.inputs, self.W, *self.theanoArgs, **self.theanoKwArgs))

class MaxPooling2D(Layer_ABC) :
	
	def __init__(self, downScaleFactors) :
		"""downScaleFactors is the factor by which to downscale vertical and horizontal dimentions. (2,2) will halve the image in each dimension."""
		self.downScaleFactors = downScaleFactors

	def _setOutputs(self) :
		self.outputs =  max_pool_2d(self.inputs, self.downScaleFactors)

class Flatten(Layer_ABC) :

	def __init__(self, outdim) :
		"""Flattens the output of a convolution to a given numer of dimentions"""
		self.outdim = outdim

	def _setOutputs(self) :
		self.outputs = T.flatten(self.inputs, self.outdim)