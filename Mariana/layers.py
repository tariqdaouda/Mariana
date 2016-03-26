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

__all__ = ["Layer_ABC", "Output_ABC", "Classifier_ABC", "Input", "Hidden", "Composite", "Embedding", "SoftmaxClassifier", "Regression", "Autoencode"]

TYPE_UNDEF_LAYER = -1
TYPE_INPUT_LAYER = 0
TYPE_HIDDEN_LAYER = 1
TYPE_OUTPUT_LAYER = 2

class Layer_ABC(object) :
	"The interface that every layer should expose"

	__metaclass__ = ABCMeta

	def __init__(self,
		size,
		saveOutputs=MSET.SAVE_OUTPUTS_DEFAULT,
		initializations=[],
		decorators=[],
		name=None,
		**kwrags
	) :

		#a unique tag associated to the layer
		self.appelido = numpy.random.random()
		if name is not None :
			self.name = name
		else :
			self.name = "%s_%s" %(self.__class__.__name__, self.appelido)

		self.type = TYPE_UNDEF_LAYER 

		self.nbInputs = None
		self.inputs = None
		self.nbOutputs = size
		self.outputs = None # this is a symbolic var
		self.saveOutputs = saveOutputs

		if saveOutputs :
			initLO = numpy.zeros(self.nbOutputs, dtype=theano.config.floatX)
			self.last_outputs = theano.shared(value = numpy.matrix(initLO)) # this will be a shared variable with the last values of outputs
		else :
			self.last_outputs = None

		self.decorators = decorators
		self.initializations = initializations

		self.feedsInto = OrderedDict()
		self.feededBy = OrderedDict()

		self.network = None
		self._inputRegistrations = set()

		self._mustInit = True
		self._decorating = False

	def getParameterDict(self) :
		"""returns the layer's parameters as dictionary"""
		from theano.compile import SharedVariable
		res = {}
		for k, v in self.__dict__.iteritems() :
			if k != 'last_outputs' and isinstance(v, SharedVariable) :
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
		
	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		raise NotImplemented("Should be implemented in child")

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
	
	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		"""Sets the output of the layer. This function is called by _init() ans should be initialized in child"""
		raise NotImplemented("Should be implemented in child")

	def _decorate(self) :
		"""applies decorators"""
		self._decorating = True
		for d in self.decorators :
			d.apply(self)
		self._decorating = False

	def _init(self) :
		"Initialize the layer making it ready for training. This function is automatically called before train/test etc..."
		if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.feededBy) ) :
			self._whateverFirstInit()
			self._initParameters()
			self._setOutputs()
			self._decorate()
			self._whateverLastInit()
			if self.outputs is None :
				raise ValueError("Invalid layer '%s' has no defined outputs" % self.name)

			for l in self.feedsInto.itervalues() :
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
		"""Connect the layer to another one. Using the '>' operator to connect to layers is actually calls this function.
		This function returns the resulting network"""
		if layer.network is None :
			layer.network = self.network
		else :
			self.network.merge(self, layer)
		self.network.addEdge(self, layer)

		layer._femaleConnect(self)
		layer.feededBy[self.name] = self
		self._maleConnect(layer)
		self.feedsInto[layer.name] = layer
		
		return self.network

	def disconnect(self, layer) :
		"""Severs a connection"""
		del(self.feedsInto[layer.name])
		del(layer.feededBy[self.name])
		self.network.removeEdge(self, layer)
		self.network._mustInit = True

	def getOutputs(self) :
		"""Return the last outputs of the layer"""
		try :
			return self.last_outputs.get_value()
		except AttributeError :
			raise AttributeError("Impossible to get the last ouputs of this layer, if you want them to be stored create with saveOutputs = True")

	def _dot_representation(self) :
		"returns the representation of the node in the graph DOT format"
		return '[label="%s: %sx%s"]' % (self.name, self.nbInputs, self.nbOutputs)

	def __gt__(self, pathOrLayer) :
		"""Alias to connect, make it possible to write things such as layer1 > layer2"""
		return self.connect(pathOrLayer)

	def __div__(self, pathOrLayer) :
		"""Alias to disconnect, make it possible to write things such as layer1 / layer2"""
		return self.disconnect(pathOrLayer)

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

	def __init__(self, size, nbDimentions, dictSize, initializations = [MI.SmallUniformEmbeddings()], learningScenario = None, name = None, **kwargs) :
		"""
		:param size int: the size of the input vector (if your input is a sentence this should be the number of words in it).
		:param nbDimentions int: the number of dimentions in wich to encode each word.
		:param dictSize int: the total number of words. 
		"""

		Layer_ABC.__init__(self, size, nbDimentions, initializations=initializations, name = name, **kwargs)
		self.network = MNET.Network()
		self.network.addInput(self)
		
		self.learningScenario = learningScenario
		self.type = TYPE_INPUT_LAYER
		self.dictSize = dictSize
		self.nbDimentions = nbDimentions
		
		self.nbInputs = size
		self.nbOutputs = self.nbDimentions*self.nbInputs
		
		self.embeddings = None
		self.inputs = tt.imatrix(name = "embInp_" + self.name)

	def getParameterShape(self, param) :
		if param == "embeddings" :
			return (self.dictSize, self.nbDimentions)
		else :
			raise ValueError("Unknow parameter: %s" % param)
	
	def getEmbeddings(self, idxs = None) :
		"""returns the embeddings.
		
		:param list idxs: if provided will return the embeddings only for those indexes 
		"""
		if idxs :
			return self.embeddings.get_value()[idxs]
		return self.embeddings.get_value()

	def _setOutputs(self) :
		self.preOutputs = self.embeddings[self.inputs]
		self.outputs = self.preOutputs.reshape((self.inputs.shape[0], self.nbOutputs))
		
	def _dot_representation(self) :
		return '[label="%s: %s" shape=invhouse]' % (self.name, self.nbOutputs)

class Input(Layer_ABC) :
	"An input layer"
	def __init__(self, size, name = None, **kwargs) :
		Layer_ABC.__init__(self, size, name = name, **kwargs)
		self.kwargs = kwargs
		self.type = TYPE_INPUT_LAYER
		self.nbInputs = size
		self.network = MNET.Network()
		self.network.addInput(self)

		self.inputs = tt.matrix(name = self.name)
	
	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = self.inputs
	
	def _femaleConnect(self, *args) :
		raise ValueError("Nothing can be connected into an input layer")

	def _dot_representation(self) :
		return '[label="%s: %s" shape=invhouse]' % (self.name, self.nbOutputs)

class Composite(Layer_ABC):
	"""A Composite layer concatenates the outputs of several other layers
	for example is we have::

		c = Composite()
		layer1 > c
		layer2 > c

	The output of c will be single vector: [layer1.output, layer2.output]
	"""
	def __init__(self, name = None):
		Layer_ABC.__init__(self, size = None, name = name)

	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = 0
		self.nbInputs += layer.nbOutputs
		self.nbOutputs = self.nbInputs
		
	def _setOutputs(self) :
		self.outputs = tt.concatenate( [l.outputs for l in self.feededBy.itervalues()], axis = 1 )

	def _dot_representation(self) :
		return '[label="%s: %s" shape=tripleoctogon]' % (self.name, self.nbOutputs)

class Hidden(Layer_ABC) :
	"A hidden layer"
	def __init__(self, size, activation = MA.ReLU(), initializations = [MI.SmallUniformWeights(), MI.ZerosBias()], learningScenario = None, name = None, regularizations = [], **kwargs) :
		Layer_ABC.__init__(self,
			size,
			name=name,
			initializations=initializations,
			**kwargs
		)

		self.activation=activation
		self.learningScenario=learningScenario
		self.W = None
		self.b = None
		
		self.regularizationObjects = regularizations
		self.regularizations = []

		self.type = TYPE_HIDDEN_LAYER
		
	def _femaleConnect(self, layer) :
		if self.nbInputs is None :
			self.nbInputs = layer.nbOutputs
		elif self.nbInputs != layer.nbOutputs :
			raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )

	def _setOutputs(self) :
		"""initialises weights and bias. By default weights are setup to random low values, use Mariana decorators
		to change this behaviour."""
		for layer in self.feededBy.itervalues() :
			if self.inputs is None :
				self.inputs = layer.outputs	
			else :
				self.inputs += layer.outputs
		
		if self.W  is None:
			raise ValueError("No initialization was defined for weights (self.W)")
		
		if self.b is None:
			MI.ZerosBias().apply(self)
			# raise ValueError("No initialization was defined for bias (self.b)")

		self.outputs = self.activation.apply(self, tt.dot(self.inputs, self.W) + self.b)
		for reg in self.regularizationObjects :
			self.regularizations.append(reg.getFormula(self))

	def getParameterShape(self, param) :
		if param == "W" :
			return (self.nbInputs, self.nbOutputs)
		elif param == "b" :
			return (self.nbOutputs,)
		else :
			raise ValueError("Unknow parameter: %s" % param)

	def getW(self) :
		"""Return the weight values"""
		return self.W.get_value()

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias and activation function.
		You can use kwargs to setup any other attribute of the new layer, just like you would for an instanciation"""
		res = self.__class__(self.nbOutputs, activation = self.activation, **kwargs)
		res.W = self.W
		res.b = self.b
		
		return res

	def cloneBare(self, **kwargs) :
		"""Same as clone() but lets you redefine any parameter other than Weights and Bias.
		This function can be very handy if you are trying to salvage old pickled layers that were created using an older version of Mariana."""
		res = self.__class__(self.nbOutputs, **kwargs)
		res.W = self.W
		res.b = self.b
		
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
		self.updates_lastOutputs = None
		self.cost = None
		self.test_cost = None
		self.updates = None

	def _maleConnect(self, *args) :
		raise ValueError("An output layer cannot be connected to something")

	def _femaleConnect(self, layer) :
		Hidden._femaleConnect(self, layer)
		self.network.addOutput(self)

	def _backTrckDependencies(self) :
		"""Finds all the hidden layers the ouput layer is influenced by"""
		def _bckTrk(deps, layer) :
			for l in layer.feededBy.itervalues() :
				if l.__class__ is not Input :
					deps[l.name] = l
					_bckTrk(deps, l)
			return deps

		self.dependencies = _bckTrk(self.dependencies, self)

	def _userInit(self) :
		"""Here you can specify custom initializations of your layer. This called just before the costs are computed. By default does nothing"""
		pass

	def _setTheanoFunctions(self) :
		"""Creates theano_train/theano_test/theano_propagate functions and calls setCustomTheanoFunctions to
		create user custom theano functions."""

		self._backTrckDependencies()
		self._userInit()

		self.cost = self.costObject.apply(self, self.targets, self.outputs, "training")
		self.test_cost = self.costObject.apply(self, self.targets, self.outputs, "testing")

		for l in self.dependencies.itervalues() :
			if l.__class__  is not Composite :
				try :
					for reg in l.regularizations :
						self.cost += reg
				except AttributeError :
					pass

		self.updates = self.learningScenario.apply(self, self.cost)

		for l in self.dependencies.itervalues() :
			try :
				updates = l.learningScenario.apply(l, self.cost)
			except AttributeError :
				updates = self.learningScenario.apply(l, self.cost)
			self.updates.extend(updates)

		self.updates_lastOutputs = []
		for l in self.network.layers.itervalues() :
			if ( l.last_outputs is not None ) and ( l.outputs is not None ) :
				self.updates.append( (l.last_outputs, l.outputs ) )
	
		self.train = MWRAP.TheanoFunction("train", self, [self.cost], { "targets" : self.targets }, updates = self.updates, allow_input_downcast=True)
		self.test = MWRAP.TheanoFunction("test", self, [self.test_cost], { "targets" : self.targets }, updates = self.updates_lastOutputs, allow_input_downcast=True)
		self.propagate = MWRAP.TheanoFunction("propagate", self, [self.outputs], updates = self.updates_lastOutputs, allow_input_downcast=True)

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

	def clone(self, **kwargs) :
		"""Returns a free output layer with the same weights, bias, activation function, learning scenario, and cost.
		You can use kwargs to setup any other attribute of the new layer, just like you would for an instanciation"""
		return Hidden.clone(self, learningScenario = self.learningScenario, costObject = self.costObject)

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
	def __init__(self, nbOutputs, learningScenario, costObject, temperature = 1, name = None, **kwargs) :
		kwargs["activation"] = MA.Softmax(temperature = temperature)
		kwargs["learningScenario"] = learningScenario
		kwargs["costObject"] = costObject
		kwargs["name"] = name
		Classifier_ABC.__init__(self, nbOutputs, **kwargs)
		self.targets = tt.ivector(name = "targets_" + self.name)
	
	def setCustomTheanoFunctions(self) :
		"""defined theano_classify, that returns the argmax of the output"""
		self.classify = MWRAP.TheanoFunction("classify", self, [ tt.argmax(self.outputs) ], updates = self.updates_lastOutputs)

	def clone(self, **kwargs) :
		"""Returns a free output layer with the same weights, bias, activation function, learning scenario, and cost.
		You can use kwargs to setup any other attribute of the new layer, just like you would for an instanciation"""
		return Hidden.cloneBare(self, learningScenario = self.learningScenario, costObject = self.costObject, **kwargs)

	def _dot_representation(self) :
		return '[label="SoftM %s: %s" shape=doublecircle]' % (self.name, self.nbOutputs)

class Regression(Output_ABC) :
	"""For regressions, works great with a mean squared error cost"""
	def __init__(self, nbOutputs, activation, learningScenario, costObject, name = None, **kwargs) :
		Output_ABC.__init__(self, nbOutputs, activation = activation, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.targets = tt.matrix(name = "targets")
	
	def _dot_representation(self) :
		return '[label="Reg %s: %s" shape=circle]' % (self.name, self.nbOutputs)

class Autoencode(Output_ABC) :
	"""An auto encoding layer. This one takes another layer as inputs and tries to reconstruct its activations.
	You could achieve the same result with a Regresison layer, but this one has the advantage of not needing to be fed specific inputs"""

	def __init__(self, layer, activation, learningScenario, costObject, name = None, **kwargs) :
		Output_ABC.__init__(self, layer.nbOutputs, activation = activation, learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.layer = layer

	def _userInit(self):
		self.targets = self.layer.outputs
	
	def setCustomTheanoFunctions(self) :
		self.train = MWRAP.TheanoFunction("train", self, [self.cost], {}, updates = self.updates, allow_input_downcast=True)
		self.test = MWRAP.TheanoFunction("test", self, [self.test_cost], {}, updates = self.updates_lastOutputs, allow_input_downcast=True)

	def _dot_representation(self) :
		return '[label="%s: AE(%s)" shape=circle]' % (self.name, self.layer.name)
