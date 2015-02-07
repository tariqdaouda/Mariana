from abc import ABCMeta#, abstractmethod

from types import *
import cPickle

import theano, numpy, time
import theano.tensor as tt
import Mariana.costs as MC
from Network import *

class TheanoFct(object) :

	def __init__(self, name, theano_fct) :
		self.theano_fct = theano_fct
		self.name = name

	def run(self, *args) :
		return self.theano_fct(*args)
	
	def __call__(self, *args) :
		return self.run(*args)

	def __repr__(self) :
		return "<Mariana Theano Fct '%s'>" % self.name

	def __str__(self) :
		return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)

class LayerABC(object) :

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
		
	def clone(self) :
		"""Returns a free layer with the same weights and bias"""
		res = self.__class__(self.nbOutputs)
		res.W = self.W
		res.b = self.b
		return res

	def _registerInput(self, inputLayer) :
		self._inputRegistrations.add(inputLayer.name)

	def _init(self) :
		if len(self._inputRegistrations) == len(self.feededBy) :
			self._setOutputs()
			if self.outputs == None :
				raise ValueError("Invalid network, layer '%s' has no defined outputs" % self.name)

			for l in self.feedsInto.itervalues() :
				if l.inputs is None :
					l.inputs = self.outputs
				else :
					l.inputs += self.outputs
				l._registerInput(self)
				l._init()

	#theano hates abstract methods defined with a decorator
	def _setOutputs(self) :
		raise NotImplemented("Should be implemented in child")

	def _setNbInputs(self, layer) :
		if self.nbInputs is not None and self.nbInputs != layer.nbOutputs :
			raise ValueError("There's already a layer of '%d' outputs feeding into %s', can't connect '%s' of '%d' outputs" % (self.nbInputs, self.name, layer.name, layer.nbOutputs))
		self.nbInputs = layer.nbOutputs
	
	def _con(self, layerOrList) :
		def _connectLayer(me, layer) :
			layer._setNbInputs(me)
			me.feedsInto[layer.name] = layer
			layer.feededBy[me.name] = me

		if self.network is None :
			self.network = Network(self)

		if type(layerOrList) is ListType :
			for p in layerOrList :
				layer = p[1].entryLayer
				_connectLayer(self, layer)
				self.network.merge(self, p[1])
				layer.network = self.network
		else :
			layer = layerOrList
			if isinstance(layer, Output) or issubclass(layer.__class__, Output) :
				self.network.addOutput(layer)

			_connectLayer(self, layer)
			self.network.addEdge(self, layer)
			layer.network = self.network
			
		return self.network

	def __gt__(self, pathOrLayer) :
		return self._con(pathOrLayer)

	def __repr__(self) :
		return "(Mariana Layer %s: %sx%s )" % (self.name, self.nbInputs, self.nbOutputs)

class Hidden(LayerABC) :

	def __init__(self, nbOutputs, activation = tt.tanh, name = None, sparsity = 0) :
		LayerABC.__init__(self, nbOutputs, name = name)
		self.activation = activation
		self.sparsity = sparsity

	def _setOutputs(self) :

		if self.W is None :
			rng = numpy.random.RandomState(int(time.time()))
			if self.activation is tt.tanh :
				initWeights = rng.uniform(
					low = -numpy.sqrt(6. / (self.nbInputs + self.nbOutputs)),
					high = numpy.sqrt(6. / (self.nbInputs + self.nbOutputs)),
					size = (self.nbInputs, self.nbOutputs)
				)
			else :
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

class Input(Hidden) :

	def __init__(self, nbInputs, name = None) :
		LayerABC.__init__(self, nbInputs, name = name)
		self.nbInputs = nbInputs
		self.network = Network(self, self)

	def _setOutputs(self) :
		self.outputs = tt.matrix(name = self.name + "_X")

	def __repr__(self) :
		return "(Mariana input %s: %s)" % (self.name, self.nbInputs)

class Output(Hidden) :

	def __init__(self, nbOutputs, activation, costFct, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
		
		Hidden.__init__(self, nbOutputs, activation = activation, name = name)
		self.costFct = costFct
		self.lr = lr
		self.l1 = l1
		self.l2 = l2
		self.momentum = momentum	
		self.y = tt.ivector(name = self.name + "_Y")
		self.dependencies = OrderedDict()

	def _backTrckDependencies(self) :
		def _bckTrk(deps, layer) :		
			for l in layer.feededBy.itervalues() :
				if l.__class__ is not Input :
					deps[l.name] = l
					_bckTrk(deps, l)
			return deps

		self.dependencies = _bckTrk(self.dependencies, self)

	def _setOutputs(self) :
		
		Hidden._setOutputs(self)

		# self.outputs = sum([l.outputs for l in self.feededBy.itervalues()])
		self._backTrckDependencies()
		for l in self.dependencies.itervalues() :
			self.params.extend(l.params)

		self.inputLayer = self.network.entryLayer
		
		L1 =  self.l1 * ( abs(self.W) + sum( [abs(l.W).sum() for l in self.dependencies.values()] ) )
		L2 = self.l2 * ( self.W**2 + sum( [(l.W**2).sum() for l in self.dependencies.values()] ) )
		self.cost = self.costFct(self.y, self.outputs) #+ L1 + L2

		self.updates = []
		for param in self.params :
			gparam = tt.grad(self.cost, param)
			momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
			self.updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
			self.updates.append((param, param - self.lr * momentum_param))
			# self.updates.append((param, param - self.lr * gparam))

		self.theano_train = theano.function(inputs = [self.inputLayer.outputs, self.y], outputs = [self.cost, self.outputs], updates = self.updates)#,  mode='DebugMode')
		self.theano_test = theano.function(inputs = [self.inputLayer.outputs, self.y], outputs = [self.cost, self.outputs])
		self.theano_propagate = theano.function(inputs = [self.inputLayer.outputs], outputs = self.outputs)
		
		self._setTheanoFunction()
	
	def _initThenoFunctions(self) :
		for k, v in self.__dict__ :
			if k.find("theano") == 0 :
				self.__dict__[k] = TheanoFct(v)

	def _setTheanoFunction(self) :
		"This is where you should put the definitions of your custom theano functions"
		pass

	def toHidden(self) :
		"returns a hidden layer with the same activation function, weights and bias"
		h = Hidden(self.nbOutputs, activation = self.activation, name = self.name, sparsity = 0)
		h.W = self.W
		h.b = self.b
		return h

class ClassifierABC(Output):
		"An abstract Classifier"
		def __init__(self, nbOutputs, activation, costFct, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
			Output.__init__(self, nbOutputs, activation,  costFct, lr, l1, l2, momentum , name)

		def _setTheanoFunction(self) :
			self.theano_predict = None

		def predict(self) :
			if self.theano_predict is None :
				raise NotImplemented("theano predict must be defined in child's _setTheanoFunction()")

class SoftmaxClassifier(ClassifierABC) :

	def __init__(self, nbOutputs, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
		ClassifierABC.__init__(self, nbOutputs, activation = tt.nnet.softmax, costFct = MC.negativeLogLikelihood, lr = lr, l1 = l1, l2 = l2, momentum = momentum, name = name)

	def _setTheanoFunction(self) :
		self.theano_predict = theano.function(inputs = [self.inputLayer.outputs], outputs = tt.argmax(self.outputs, axis = 1))
