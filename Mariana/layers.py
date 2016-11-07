from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

# import types
import theano, numpy, time
import theano.tensor as tt

import Mariana.activations as MA
import Mariana.initializations as MI
import Mariana.settings as MSET

import Mariana.network as MNET
import Mariana.wrappers as MWRAP
import Mariana.candies as MCAN

__all__ = ["Layer_ABC", "WeightBiasOutput_ABC", "WeightBias_ABC", "Output_ABC", "Input", "Hidden", "Composite", "Embedding", "SoftmaxClassifier", "Regression", "Autoencode"]

class Layer_ABC(object) :
	"The interface that every layer should expose"

	__metaclass__ = ABCMeta

	def __new__(cls, *args, **kwargs) :
		obj = super(Layer_ABC, cls).__new__(cls, *args, **kwargs)
		obj.creationArguments = {
			"args": args,
			"kwargs": kwargs,
		}

		return obj

	def __init__(self,
		size,
		layerType,
		activation = MA.Pass(),
		regularizations = [],
		initializations=[],
		learningScenario=None,
		decorators=[],
		name=None
	):

		self.isLayer = True

		#a unique tag associated to the layer
		self.appelido = numpy.random.random()

		if name is not None :
			self.name = name
		else :
			self.name = "%s_%s" %(self.__class__.__name__, self.appelido)

		self.type = layerType

		self.nbInputs = None
		self.inputs = None
		self.nbOutputs = size
		self.outputs = None # this is a symbolic var
		self.testOutputs = None # this is a symbolic var

		self.preactivation_outputs = None
		self.preactivation_testOutputs = None

		self.activation = activation
		self.regularizationObjects = regularizations
		self.regularizations = []
		self.decorators = decorators
		self.initializations = initializations
		self.learningScenario=learningScenario

		self.network = MNET.Network()
		self.network._addLayer(self)

		self._inputRegistrations = set()

		self._mustInit = True
		self._mustReset = True
		self._decorating = False

	def getParameterDict(self) :
		"""returns the layer's parameters as dictionary"""
		from theano.compile import SharedVariable
		res = {}
		for k, v in self.__dict__.iteritems() :
			if isinstance(v, SharedVariable) :
				res[k] = v
		return res

	def getParameters(self) :
		"""returns the layer's parameters"""
		return self.getParameterDict().values()

	def getParameterNames(self) :
		"""returns the layer's parameters names"""
		return self.getParameterDict().keys()

	def getParameterShape(self, param) :
		"""Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
		raise NotImplemented("Should be implemented in child")

	def getOutputShape(self):
		"""returns the shape of the outputs"""
		return (self.nbOutputs, )

	def clone(self, reset = False) :
		"""Returns a free layer with the same parameters"""
		newLayer = self.__class__(*self.creationArguments["args"], **self.creationArguments["kwargs"])
		for k, v in self.getParameterDict().iteritems() :
			setattr(newLayer, k, v)
		return newLayer

	def _registerInput(self, inputLayer) :
		"Registers a layer as an input to self. This function is first called by input layers. Initialization can only start once all input layers have been registered"
		self._inputRegistrations.add(inputLayer.name)

	def _whateverFirstInit(self) :
		"""The first function called during initialization. Does nothing by default, you can put in it
		whatever pre-action you want performed on the layer prior to normal initialization"""
		pass

	def _whateverLastInit(self) :
		"""The last function called during initialization. Does nothing by default, you can put in it
		whatever post-action you want performed on the layer post normal initialization"""
		pass

	def _initParameters(self) :
		"""creates the parameters if necessary"""
		for init in self.initializations :
			init.apply(self)

	def initParameter(self, parameter, value) :
		"""Initialize a parameter, raise value error if already initialized"""
		if not hasattr(self, parameter) or getattr(self, parameter) is None :
			setattr(self, parameter, value)
		else :
			raise ValueError("Parameter '%s' or layer '%s' has already been initialized" % (parameter, self.name) )

	def updateParameter(self, parameter, value) :
		"""Update the value of an already initialized parameter. Raise value error if the parameter has not been initialized"""
		if k not in self.getParameterDict().keys() :
			raise ValueError("Parameter '%s' has not been initialized as parameter of layer '%s'" % (parameter, self.name) )
		else :
			setattr(self, parameter, value)

	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		"""Defines the outputs and testOutputs of the layer before the application of the activation function. This function is called by _init() ans should be written in child."""
		raise NotImplemented("Should be implemented in child")

	def _decorate(self) :
		"""applies decorators"""
		for d in self.decorators :
			d.apply(self)

	def _activate(self) :
		"""applies activation"""
		self.preactivation_outputs = self.outputs
		self.preactivation_testOutputs = self.testOutputs

		self.outputs = self.activation.apply(self, self.outputs, 'training')
		self.testOutputs = self.activation.apply(self, self.testOutputs, 'testing')

	def _listRegularizations(self) :
		for reg in self.regularizationObjects :
			self.regularizations.append(reg.getFormula(self))

	def _setTheanoFunctions(self) :
		"""Creates propagate/propagateTest theano function that returns the layer's outputs.
		propagateTest returns the testOutputs, some decorators might not be applied.
		This is called after decorating"""
		self.propagate = MWRAP.TheanoFunction("propagate", self, [("outputs", self.outputs)], allow_input_downcast=True)
		self.propagateTest = MWRAP.TheanoFunction("propagateTest", self, [("outputs", self.testOutputs)], allow_input_downcast=True)

	def setCustomTheanoFunctions(self) :
		"""This is where you should put the definitions of your custom theano functions. Theano functions
		must be declared as self attributes using a wrappers.TheanoFunction object, cf. wrappers documentation.
		This is called just before _whateverLastInit.
		"""
		pass

	def _init(self) :
		"Initialize the layer making it ready for training. This function is automatically called before train/test etc..."
		if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.network.inConnections[self]) ) :
			if self._mustReset :
				self._whateverFirstInit()
				self._initParameters()
				self._mustReset = False

			self._setOutputs()
			self._activate()
			self._listRegularizations()
			self._decorate()
			self._setTheanoFunctions()
			self.setCustomTheanoFunctions()
			self._whateverLastInit()

			if self.outputs is None :
				raise ValueError("Invalid layer '%s' has no defined outputs" % self.name)

			for l in self.network.outConnections[self] :
				l._registerInput(self)
				l._init()
			self._mustInit = False

	def _maleConnect(self, layer) :
		"""What happens to A when A > B"""
		pass

	def _femaleConnect(self, layer) :
		"""What happens to B when A > B"""
		pass

	def connect(self, layer) :
		"""Connect the layer to another one. Using the '>' operator to connect two layers actually calls this function.
		This function returns the resulting network"""
		self.network.merge(self, layer)

		layer._femaleConnect(self)
		self._maleConnect(layer)

		return self.network

	def _dot_representation(self) :
		"returns the representation of the node in the graph DOT format"
		return '[label="%s: %s"]' % (self.name, self.getOutputShape())

	def __gt__(self, pathOrLayer) :
		"""Alias to connect, make it possible to write things such as layer1 > layer2"""
		return self.connect(pathOrLayer)

	def __repr__(self) :
		return "(Mariana %s '%s': %sx%s )" % (self.__class__.__name__, self.name, self.nbInputs, self.nbOutputs)

	def __len__(self) :
		return self.nbOutputs

	def __setattr__(self, k, v) :

		try :
			deco = self._decorating
		except AttributeError:
			object.__setattr__(self, k, v)
			return

		if deco :
			var = getattr(self, k)
			try :
				var.set_value(numpy.asarray(v, dtype=theano.config.floatX), borrow = True)
				return
			except AttributeError :
				pass

		object.__setattr__(self, k, v)

class Embedding(Layer_ABC) :
	"""This input layer will take care of creating the embeddings and training them. Embeddings are learned representations
	of the inputs that are much loved in NLP."""

	def __init__(self, size, nbDimentions, dictSize, zeroForNull = False, initializations = [MI.SmallUniformEmbeddings()], **kwargs) :
		"""
		:param int size: the size of the input vector (if your input is a sentence this should be the number of words in it).
		:param int nbDimentions: the number of dimentions in wich to encode each word.
		:param int dictSize: the total number of words.
		:param bool zeroForNull: if True the dictionnary will be augmented by one elements at te begining (index = 0) whose parameters will always be vector of zeros.
		This can be used to selectively mask some words in the input, but keep in mind that the index for the first word is moved to one.
		"""

		super(Embedding, self).__init__(size, layerType=MNET.TYPE_INPUT_LAYER,  initializations=initializations, **kwargs)

		self.zeroForNull = zeroForNull

		self.dictSize = dictSize
		self.nbDimentions = nbDimentions

		self.nbInputs = size
		self.nbOutputs = self.nbDimentions*self.nbInputs

		self.embeddings = None
		self.fullEmbeddings = None

		self.inputs = tt.imatrix(name = "embInp_" + self.name)


	def getParameterShape(self, param) :
		if param == "embeddings" :
			return (self.dictSize, self.nbDimentions)
		else :
			raise ValueError("Unknown parameter: %s" % param)

	def getEmbeddings(self, idxs = None) :
		"""returns the embeddings.

		:param list idxs: if provided will return the embeddings only for those indexes
		"""
		if not self.fullEmbeddings :
			raise ValueError("It looks like the network has not been initialized yet. Try calling self.network.init() first.")

		try :
			fct = self.fullEmbeddings.get_value
		except AttributeError :
			fct = self.fullEmbeddings.eval

		if idxs :
			return fct()[idxs]
		return fct()

	def _setOutputs(self) :
		if self.zeroForNull :
			self.null = numpy.zeros((1, self.nbDimentions))
			self.fullEmbeddings = tt.concatenate( [self.null, self.embeddings], axis = 0 )
		else :
			self.fullEmbeddings = self.embeddings
			del(self.embeddings)

		self.preOutputs = self.fullEmbeddings[self.inputs]

		self.outputs = self.preOutputs.reshape((self.inputs.shape[0], self.nbOutputs))
		self.testOutputs = self.preOutputs.reshape((self.inputs.shape[0], self.nbOutputs))

class Input(Layer_ABC) :
	"An input layer"
	def __init__(self, size, name = None, **kwargs) :
		super(Input, self).__init__(size, layerType=MNET.TYPE_INPUT_LAYER, name=name, **kwargs)
		self.kwargs = kwargs
		self.nbInputs = size

		self.inputs = tt.matrix(name = self.name)

	def _setOutputs(self) :
		"initializes the output to be the same as the inputs"
		self.outputs = self.inputs
		self.testOutputs = self.inputs

	def _femaleConnect(self, *args) :
		raise ValueError("Nothing can be connected into an input layer")

class Composite(Layer_ABC):
	"""A Composite layer concatenates the outputs of several other layers
	for example is we have::

		c = Composite()
		layer1 > c
		layer2 > c

	The output of c will be single vector: [layer1.output, layer2.output]
	"""
	def __init__(self, name = None, **kwargs):
		super(Composite, self).__init__(layerType=MNET.TYPE_HIDDEN_LAYER, size=None, name = name, **kwargs)

	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = 0
		self.nbInputs += layer.nbOutputs
		self.nbOutputs = self.nbInputs

	def _setOutputs(self) :
		outs = []
		for l in self.network.inConnections[self] :
			outs.append(l.outputs)

		self.outputs = tt.concatenate( outs, axis = 1 )
		self.testOutputs = tt.concatenate( outs, axis = 1 )

class Pass(Layer_ABC) :
	def __init__(self, name = None, **kwargs):
		super(Pass, self).__init__(layerType=MNET.TYPE_HIDDEN_LAYER, size=None, name=name, **kwargs)

	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = layer.nbOutputs
		elif self.nbInputs != layer.nbOutputs :
			raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )

	def _setOutputs(self) :
		for layer in self.network.inConnections[self] :
			if self.inputs is None :
				self.inputs = layer.outputs
			else :
				self.inputs += layer.outputs

		self.outputs = self.inputs
		self.testOutputs = self.inputs

class WeightBias_ABC(Layer_ABC) :
	"""A layer with weigth and bias. If would like to disable either one of them simply do not initialize"""

	def __init__(self, size, layerType, initializations = [MI.SmallUniformWeights(), MI.ZerosBias()], **kwargs) :
		super(WeightBias_ABC, self).__init__(size, layerType=layerType, initializations=initializations, **kwargs)

		self.W = None
		self.b = None

	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = layer.nbOutputs
		elif self.nbInputs != layer.nbOutputs :
			raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )

	def _setInputs(self) :
		"""Adds up the outputs of all incoming layers"""
		for layer in self.network.inConnections[self] :
			if self.inputs is None :
				self.inputs = layer.outputs
			else :
				self.inputs += layer.outputs

	def _setOutputs(self) :
		"""Defines, self.outputs and self.testOutputs"""
		self._setInputs()

		self.outputs = 0
		self.testOutputs = 0

		if self.W is not None:
			self.outputs = tt.dot(self.inputs, self.W)
			self.testOutputs = tt.dot(self.inputs, self.W)

		if self.b is not None:
			self.outputs = self.outputs + self.b
			self.testOutputs = self.testOutputs + self.b

	def getParameterShape(self, param) :
		if param == "W" :
			return (self.nbInputs, self.nbOutputs)
		elif param == "b" :
			return (self.nbOutputs,)
		else :
			raise ValueError("Unknown parameter: %s" % param)

	def getW(self) :
		"""Return the weight values"""
		try :
			return self.W.get_value()
		except AttributeError :
			raise ValueError("It looks like the network has not been initialized yet")

	def getb(self) :
		"""Return the bias values"""
		try :
			return self.b.get_value()
		except AttributeError :
			raise ValueError("It looks like the network has not been initialized yet")

class Hidden(WeightBias_ABC) :
	"A hidden layer with weigth and bias"
	def __init__(self, size, **kwargs) :
		super(Hidden, self).__init__(size, layerType=MNET.TYPE_HIDDEN_LAYER, **kwargs)

class Output_ABC(Layer_ABC) :
	"""The interface that every output layer should expose. This interface also provides the model functions::

		* train: upadates the parameters and returns the cost
		* test: returns the cost, ignores trainOnly decoartors
		"""

	def __init__(self, size, costObject, **kwargs) :
		super(Output_ABC, self).__init__(size, layerType=MNET.TYPE_OUTPUT_LAYER, **kwargs)
		self.type = MNET.TYPE_OUTPUT_LAYER
		self.targets = None
		self.dependencies = OrderedDict()
		self.costObject = costObject

		self.cost = None
		self.testCost = None
		self.updates = None

	def _backTrckDependencies(self) :
		"""Finds all the hidden layers the ouput layer is influenced by"""
		self.dependencies = {}
		def _bckTrk(deps, layer) :
			for l in self.network.inConnections[layer] :
				deps[l.name] = l
				_bckTrk(deps, l)
			return deps

		self.dependencies = _bckTrk(self.dependencies, self)

	def setCustomTheanoFunctions(self) :
		"""Adds train, test, model functions::

			* train: update parameters and return cost
			* test: do not update parameters and return cost without adding regularizations
		"""

		self._backTrckDependencies()
		self.cost = self.costObject.apply(self, self.targets, self.outputs, "training")
		self.testCost = self.costObject.apply(self, self.targets, self.outputs, "testing")

		for l in self.dependencies.itervalues() :
			if l.__class__  is not Composite :
				try :
					for reg in l.regularizations :
						self.cost += reg
				except AttributeError :
					pass

		self.updates = self.learningScenario.apply(self, self.cost)
		for l in self.dependencies.itervalues() :
			if l.learningScenario is not None :
				updates = l.learningScenario.apply(l, self.cost)
			else :
				updates = self.learningScenario.apply(l, self.cost)
			self.updates.extend(updates)

		self.train = MWRAP.TheanoFunction("train", self, [("score", self.cost)], { "targets" : self.targets }, updates = self.updates, allow_input_downcast=True)
		self.test = MWRAP.TheanoFunction("test", self, [("score", self.testCost)], { "targets" : self.targets }, allow_input_downcast=True)

class WeightBiasOutput_ABC(Output_ABC, WeightBias_ABC):
	"""Generic output layer with weight and bias"""
	def __init__(self, nbOutputs, costObject, learningScenario, activation, **kwargs):
		super(WeightBiasOutput_ABC, self).__init__(size=nbOutputs, costObject=costObject, learningScenario=learningScenario, activation=activation, **kwargs)

class SoftmaxClassifier(WeightBiasOutput_ABC) :
	"""A softmax (probabilistic) Classifier"""
	def __init__(self, nbOutputs, costObject, learningScenario, temperature = 1, **kwargs) :
		super(SoftmaxClassifier, self).__init__(nbOutputs, costObject=costObject, learningScenario=learningScenario, activation=MA.Softmax(temperature=temperature), **kwargs)

	def setCustomTheanoFunctions(self) :
		"""defines::

			* classify: return the argmax of the outputs applying all the decorators.
			* predict: return the argmax of the test outputs (some decorators may not be applied).
			* classificationAccuracy: returns the accuracy (between [0, 1]) of the model, computed on outputs.
			* predictionAccuracy: returns the accuracy (between [0, 1]) of the model, computed on test outputs.
		"""
		Output_ABC.setCustomTheanoFunctions(self)
		clas = tt.argmax(self.outputs, axis=1)
		pred = tt.argmax(self.testOutputs, axis=1)

		self.classify = MWRAP.TheanoFunction("classify", self, [ ("class", clas) ], allow_input_downcast=True)
		self.predict = MWRAP.TheanoFunction("predict", self, [ ("class", pred) ], allow_input_downcast=True)

		clasAcc = tt.mean( tt.eq(self.targets, clas ) )
		predAcc = tt.mean( tt.eq(self.targets, pred ) )

		self.classificationAccuracy = MWRAP.TheanoFunction("classificationAccuracy", self, [("accuracy", clasAcc)], { "targets" : self.targets }, allow_input_downcast=True)
		self.predictionAccuracy = MWRAP.TheanoFunction("predictionAccuracy", self, [("accuracy", predAcc)], { "targets" : self.targets }, allow_input_downcast=True)

		self.trainAndAccuracy = MWRAP.TheanoFunction("trainAndAccuracy", self, [("score", self.cost), ("accuracy", clasAcc)], { "targets" : self.targets },  updates = self.updates, allow_input_downcast=True)
		self.testAndAccuracy = MWRAP.TheanoFunction("testAndAccuracy", self, [("score", self.testCost), ("accuracy", predAcc)], { "targets" : self.targets }, allow_input_downcast=True)

class Regression(WeightBiasOutput_ABC) :
	"""For regressions, works great with a mean squared error cost"""
	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		super(Regression, self).__init__(nbOutputs, activation=activation, learningScenario=learningScenario, costObject=costObject, name=name, **kwargs)
		self.targets = tt.matrix(name = "targets")

class Autoencode(WeightBiasOutput_ABC) :
	"""An auto encoding layer. This one takes another layer as inputs and tries to reconstruct its activations.
	You could achieve the same result with a Regression layer, but this one has the advantage of not needing to be fed specific inputs"""

	def __init__(self, targetLayerName, activation, learningScenario, costObject, name=None, **kwargs) :
		super(Autoencode, self).__init__(None, activation=activation, learningScenario=learningScenario, costObject=costObject, name=name, **kwargs)
		self.targetLayerName = targetLayerName

	def _whateverFirstInit(self) :
		self.nbOutputs = self.network[self.targetLayerName].nbOutputs
		self.targets = self.network[self.targetLayerName].outputs

	def setCustomTheanoFunctions(self) :
		super(Autoencode, self).setCustomTheanoFunctions()
		self.train = MWRAP.TheanoFunction("train", self, [("score", self.cost)], {}, updates = self.updates, allow_input_downcast=True)
		self.test = MWRAP.TheanoFunction("test", self, [("score", self.testCost)], {}, allow_input_downcast=True)
