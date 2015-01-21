from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

from types import *
import cPickle

import theano, numpy
import theano.tensor as tt
import Mariana.costs as MC


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
		return "(%s: %sx%s )" % (self.name, self.nbInputs, self.nbOutputs)

class Hidden(LayerABC) :

	def __init__(self, nbOutputs, activation = tt.tanh, name = None) :
		LayerABC.__init__(self, nbOutputs, name = name)
		self.activation = activation

	def _setOutputs(self) :
		# print "====", self, self.inputs
		initWeights = numpy.random.random((self.nbInputs, self.nbOutputs))
		initWeights = (initWeights/sum(initWeights))
		initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		self.W = theano.shared(value = initWeights, name = self.name + "_W")
			
		initBias = numpy.zeros((self.nbOutputs,), dtype=theano.config.floatX)
		self.b = theano.shared(value = initBias, name = self.name + "_b")

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

		self._setTheanoFunction()

	def _setTheanoFunction(self) :
		self.theano_train = theano.function(inputs = [self.inputLayer.outputs, self.y], outputs = [self.cost, self.outputs], updates = self.updates)#,  mode='DebugMode')
		self.theano_test = theano.function(inputs = [self.inputLayer.outputs, self.y], outputs = [self.cost, self.outputs])
		self.theano_propagate = theano.function(inputs = [self.inputLayer.outputs], outputs = self.outputs)
		# print theano.printing.debugprint(self.theano_train)
		# stop

class SoftmaxClassifier(Output) :

	def __init__(self, nbOutputs, lr = 0.1, l1 = 0, l2 = 0, momentum = 0, name = None) :
		Output.__init__(self, nbOutputs,  costFct = MC.negativeLogLikelihood, activation = tt.nnet.softmax, lr = lr, l1 = l1, l2 = l2, momentum = momentum, name = name)

	def _setTheanoFunction(self) :
		Output._setTheanoFunction(self)
		self.theano_predict = theano.function(inputs = [self.inputLayer.outputs], outputs = tt.argmax(self.outputs, axis = 1))

class Network(object) :

	def __init__(self, entryLayer, inputLayer = None) :
		
		self.entryLayer = entryLayer
		self.inputLayer = inputLayer
		self.layers = OrderedDict()
		self.outputs = OrderedDict()
		self.edges = set()

		self.params = []

		self._mustInit = True

	def addParams(self, params) :
		self.params.extend(params)

	def addEdge(self, layer1, layer2) :
		self.layers[layer1.name] = layer1
		self.layers[layer2.name] = layer2
		self.edges.add( (layer1.name, layer2.name))

	def addOutput(self, ou) :
		self.outputs[ou.name] = ou

	def merge(self, conLayer, network) :
		if network.inputLayer is not None and network.inputLayer is not self.inputLayer :
			raise ValueError("Can't merge, the network already has an input layer")

		self.addEdge(conLayer, network.entryLayer)
		
		network.entryLayer = self.entryLayer
		self.outputs.update(network.outputs)
		self.layers.update(network.layers)
		self.edges = self.edges.union(network.edges)

	def _init(self) :
		if self._mustInit :
			self.entryLayer._init()
			self._mustInit = False

	def train(self, x, y) :
		self._init()
		ret = {}
		for o in self.outputs.itervalues() :
			ret[o.name] = o.theano_train(x, y)
		return ret

	def test(self, x) :
		self._init()
		ret = {}
		for o in self.outputs.itervalues() :
			ret[o.name] = o.theano_test(x)
		return ret

	def propagate(self, x) :
		self._init()
		ret = {}
		for o in self.outputs.itervalues() :
			ret[o.name] = o.theano_propagate(x)
		return ret

	def predict(self, x) :
		self._init()
		ret = {}
		for o in self.outputs.itervalues() :
			ret[o.name] = o.theano_predict(x)
		return ret

	def __str__(self) :
		s = []
		for o in self.outputs :
			s.append(o)

		return "Net: %s > ... > [%s]" % (self.inputLayer.name, ', '.join(s))
