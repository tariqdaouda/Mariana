from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

# import types
import theano, numpy, time
import theano.tensor as tt

import Mariana.activations as MA
import Mariana.settings as MSET

import Mariana.network as MNET
import Mariana.wrappers as MWRAP

__all__ = ["Layer_ABC", "Output_ABC", "Classifier_ABC", "Input", "Hidden", "Composite", "SoftmaxClassifier", "Regression", "Autoencode", "Embedding"]

TYPE_UNDEF_LAYER = -1
TYPE_INPUT_LAYER = 0
TYPE_OUTPUT_LAYER = 1
TYPE_HIDDEN_LAYER = 1

class Layer_ABC(object) :
	"The interface that every layer should expose"

	__metaclass__ = ABCMeta

	def __init__(self,
		size,
		saveOutputs=MSET.SAVE_OUTPUTS_DEFAULT,
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

		if saveOutputs :
			initLO = numpy.zeros(self.nbOutputs, dtype=theano.config.floatX)
			self.last_outputs = theano.shared(value = numpy.matrix(initLO)) # this will be a shared variable with the last values of outputs
		else :
			self.last_outputs = None

		self.decorators = decorators

		self.feedsInto = OrderedDict()
		self.feededBy = OrderedDict()

		self.network = None
		self._inputRegistrations = set()

		self._mustInit = True
		self._decorating = False

	def addDecorator(self, decorator) :
		"""Add a decorator to the layer"""
		self.decorators.append(decorator)

	def getParams(self) :
		"""returns the layer parameters"""
		raise NotImplemented("Should be implemented in child")

	def getSubtensorParams(self) :
		"""theano has a special optimisation for when you want to update just a subset of a tensor (matrix). Use this function to return a List
		of tuples: (tensor, subset). By default it returns an empty list"""
		return []

	def clone(self, **kwargs) :
		"""Returns a free layer with the same weights and bias. You can use kwargs to setup any attribute of the new layer"""
		raise NotImplemented("Should be implemented in child")

	def _registerInput(self, inputLayer) :
		"Registers a layer as an input to self. This function is first called by input layers. Initialisation can only start once all input layers have been registered"
		self._inputRegistrations.add(inputLayer.name)

	def _init(self) :
		"Initialise the layer making it ready for training. This function is automatically called before train/test etc..."
		if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.feededBy) ) :
			self._setOutputs()
			self._decorate()
			if self.outputs is None :
				raise ValueError("Invalid layer '%s' has no defined outputs" % self.name)

			for l in self.feedsInto.itervalues() :
				l._registerInput(self)
				l._init()
			self._mustInit = False

	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		"""Sets the output of the layer. This function is called by _init() ans should be initialised in child"""
		raise NotImplemented("Should be implemented in child")

	def _decorate(self) :
		"""applies decorators"""
		self._decorating = True
		for d in self.decorators :
			d.decorate(self)
		self._decorating = False

	def connect(self, layer) :
		"""Connect the layer to another one. Using the '>' operator to connect to layers is actually calls this function.
		This function returns the resulting network"""
		def _connectLayer(me, layer) :
			me.feedsInto[layer.name] = layer
			layer.feededBy[me.name] = me
			
			me.network.addEdge(me, layer)
			if layer.network is not None :
				me.network.merge(me, layer)
			layer.network = me.network

		layer = layer
		if isinstance(layer, Input) :
			raise ValueError("Nothing can be connected to an input layer")

		if isinstance(layer, Output_ABC) or issubclass(layer.__class__, Output_ABC) :
			self.network.addOutput(layer)
		_connectLayer(self, layer)

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

	def __init__(self, size, nbDimentions, dictSize, learningScenario = None, name = None, **kwargs) :
		"""
		:param size int: the size of the input vector (if your input is a sentence this should be the number of words in it).
		:param nbDimentions int: the number of dimentions in wich to encode each word.
		:param dictSize int: the total number of words. 
		"""
		Layer_ABC.__init__(self, size, nbDimentions, name = name, **kwargs)
		self.network = MNET.Network()
		self.network.addInput(self)
		
		self.learningScenario = learningScenario
		self.type = TYPE_INPUT_LAYER
		self.dictSize = dictSize
		self.nbDimentions = nbDimentions
		
		self.nbInputs = size
		self.nbOutputs = self.nbDimentions*self.nbInputs
		
		initEmb = numpy.asarray(numpy.random.random((self.dictSize, self.nbDimentions)), dtype=theano.config.floatX)
		
		self.embeddings = theano.shared(initEmb)
		self.inputs = tt.imatrix(name = "embInp_" + self.name)

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

	def getParams(self) :
		"""returns nothing"""
		return []

	def getSubtensorParams(self) :
		"""returns the subset corresponding to the embedding"""
		return [(self.embeddings, self.preOutputs)]

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
	
	def getParams(self) :
		"""return nothing"""
		return []

	def _setOutputs(self) :
		"initialises the output to be the same as the inputs"
		self.outputs = self.inputs
		

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

	def getParams(self) :
		return []

	def _setOutputs(self) :
		for layer in self.feededBy.itervalues() :
			if self.nbInputs is None :
				self.nbInputs = 0
			self.nbInputs += layer.nbOutputs

		self.nbOutputs = self.nbInputs
		self.outputs = tt.concatenate( [l.outputs for l in self.feededBy.itervalues()], axis = 1 )

	def _dot_representation(self) :
		return '[label="%s: %s" shape=tripleoctogon]' % (self.name, self.nbOutputs)

class Hidden(Layer_ABC) :
	"A hidden layer"
	def __init__(self, size, activation = MA.ReLU(), learningScenario = None, name = None, regularizations = [], **kwargs) :
		Layer_ABC.__init__(self,
			size,
			name=name,
			**kwargs
		)

		self.activation=activation
		self.learningScenario=learningScenario
		self.W = None
		self.b = None

		self.regularizationObjects = regularizations
		self.regularizations = []

		self.type = TYPE_HIDDEN_LAYER
		
	def _setOutputs(self) :
		"""initialises weights and bias. By default weights are setup to random low values, use Mariana decorators
		to change this behaviour."""
		from theano.tensor.var import TensorVariable
		from theano.compile import SharedVariable

		for inputLayer in self.feededBy.itervalues() :
			if self.nbInputs is None :
				self.nbInputs = inputLayer.nbOutputs
				self.inputs = inputLayer.outputs
			elif self.nbInputs != inputLayer.nbOutputs :
				raise ValueError("Input size to %s as previously been set to: %s. But %s has %s outputs" % (self.name, self.nbInputs, inputLayer.name, inputLayer.nbOutputs))
			else :
				self.inputs += inputLayer.outputs
			
		if self.W is None :
			initWeights = numpy.random.random((self.nbInputs, self.nbOutputs))
			initWeights = initWeights/sum(initWeights)
			initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
			# initWeights = numpy.random.normal(0, 0.01, (self.nbInputs, self.nbOutputs))
			
			self.W = theano.shared(value = initWeights, name = "W_" + self.name)
		elif isinstance(self.W, SharedVariable) :
			if self.W.get_value().shape != (self.nbInputs, self.nbOutputs) :
				raise ValueError("weights have shape %s, but the layer has %s inputs and %s outputs" % (self.W.get_value().shape, self.nbInputs, self.nbOutputs))
		elif isinstance(self.W, numpy.ndarray) :
			if self.W.shape != (self.nbInputs, self.nbOutputs) :
				raise ValueError("weights have shape %s, but the layer has %s inputs and %s outputs" % (self.W.shape, self.nbInputs, self.nbOutputs))
			self.W = theano.shared(value = self.W, name = "W_" + self.name)
		else :
			raise ValueError("Weights should be a numpy array or a theano SharedVariable, got: %s" % type(self.W))

		if self.b is None :
			initBias = numpy.zeros((self.nbOutputs,), dtype=theano.config.floatX)
			self.b = theano.shared(value = initBias, name = "b_" + self.name)
		elif isinstance(self.b, SharedVariable) :
			if self.b.get_value().shape[0] != self.nbOutputs :
				raise ValueError("bias has a length of %s, but there are %s outputs" % (self.b.get_value().shape[0], self.nbOutputs))
		elif isinstance(self.b, numpy.ndarray) :
			if self.b.shape != self.nbOutputs :
				raise ValueError("bias has a length of %s, but there are %s outputs" % (self.b.shape[0], self.nbOutputs))
			self.b = theano.shared(value = self.b, name = "b_" + self.name)
		else :
			raise ValueError("Bias should be a numpy array or a theano SharedVariable, got: %s" % type(self.b))

		# self.outputs = theano.printing.Print('this is a very important value for %s' % self.name)(self.activation(tt.dot(self.inputs, self.W) + self.b))
		self.outputs = self.activation.function(tt.dot(self.inputs, self.W) + self.b)
		

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
		self.updates_lastOutputs = None
		self.cost = None
		self.test_cost = None
		self.updates = None

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
		"""Here you can specify custom intialisations of your layer. This called just before the costs are computed. By default does nothing"""
		pass

	def _setTheanoFunctions(self) :
		"""Creates theano_train/theano_test/theano_propagate functions and calls setCustomTheanoFunctions to
		create user custom theano functions."""

		self._backTrckDependencies()
		self._userInit()

		self.cost = self.costObject.costFct(self.targets, self.outputs)
		self.test_cost = self.costObject.costFct(self.targets, self.outputs)

		for l in self.dependencies.itervalues() :
			if l.__class__  is not Composite :
				try :
					for reg in l.regularizations :
						self.cost += reg
				except AttributeError :
					pass

		self.updates = self.learningScenario.getUpdates(self, self.cost)

		for l in self.dependencies.itervalues() :
			try :
				self.updates.extend(l.learningScenario.getUpdates(l, self.cost))
			except AttributeError :
				self.updates.extend(self.learningScenario.getUpdates(l, self.cost))

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
		Classifier_ABC.__init__(self, nbOutputs, activation = MA.Softmax(temperature = temperature), learningScenario = learningScenario, costObject = costObject, name = name, **kwargs)
		self.targets = tt.ivector(name = "targets_" + self.name)
	
	def setCustomTheanoFunctions(self) :
		"""defined theano_classify, that returns the argmax of the output"""
		self.classify = MWRAP.TheanoFunction("classify", self, [ tt.argmax(self.outputs) ], updates = self.updates_lastOutputs)

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
