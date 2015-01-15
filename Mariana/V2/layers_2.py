from abc import ABCMeta#, abstractmethod
from types import *
import cPickle

import theano, numpy
import theano.tensor as tt

class Network(object) :

	def __init__(self, entryLayer, inputLayer = None) :
		
		self.entryLayer = entryLayer
		self.inputLayer = inputLayer
		self.layers = {}
		self.outputs = {}
		self.edges = set()

		self.params = []

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

	def train(self) :
		for l in self.layers.itervalues() :
			l._init()

	def __str__(self) :
		s = []
		for o in self.outputs :
			s.append(o)

		return "Net: %s > ... > [%s]" % (self.inputLayer.name, ', '.join(s))

class LayerABC(object) :

	__metaclass__ = ABCMeta

	def __init__(self, name, nbOutputs) :
		
		self.name = name

		self.nbInputs = None
		self.inputs = None
		self.nbOutputs = nbOutputs
		self.outputs = None

		self.feedsInto = {}
		self.feededBy = {}

		self.network = None

	#theano hates abstract methods defined with a decorator
	def _init(self) :
		raise NotImplemented("Should be implemented in child")

	def _setInputs(self, layer) :
		if self.nbInputs is not None and self.nbInputs != layer.nbOutputs :
			raise ValueError("There's already a layer of '%d' outputs feeding into %s', can't connect '%s' of '%d' outputs" % (self.nbInputs, self.name, layer.name, layer.nbOutputs))
		self.nbInputs = layer.nbOutputs
		self.inputs = layer.outputs

	def _con(self, layerOrList) :
		if self.network is None :
			self.network = Network(self)

		if type(layerOrList) is ListType :
			for p in layerOrList :
				layer = p[1].entryLayer
				layer._setInputs(self)
				self.network.merge(self, p[1])
		else :
			layer = layerOrList
			layer._setInputs(self)
			
			if layer.__class__ is Output :
				self.network.addOutput(layer)
			
			self.network.addEdge(self, layer)
			layer.network = self.network

		return self.network

	def __gt__(self, pathOrLayer) :
		return self._con(pathOrLayer)

	def __repr__(self) :
		return "(%s: %sx%s )" % (self.name, self.nbInputs, self.nbOutputs)

class Hidden(LayerABC) :

	def _init(self) :
		self.activation = None

		print ">", self.name, self, self.nbInputs, self.nbOutputs
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
			self.outputs = self.activation(tt.dot(self.inputs, self.W) + self.b)

class Input(LayerABC) :

	def __init__(self, name, nbInputs) :
		LayerABC.__init__(self, name, nbInputs)
		
		self.nbInputs = nbInputs
		self.network = Network(self, self)
		self.outputs = tt.matrix(name = self.name + "_X")

	def _init(self) :
		pass

class Output(LayerABC) :

	def __init__(self, name, nbOutputs) :
		LayerABC.__init__(self, name, nbOutputs)

	def _init(self) :
		pass

def t1() :
	l = Hidden( 'h1', 3)
	# p1 = Input('i', 6) > l > Hidden( 'h2', 3)  > Output( 'o', 3)
	# p2 = Input('i', 6) > l > Hidden( 'h3', 3) > Output( 'o2', 3)

	# print '---'
	# print p1
	# print p2

def t2() :
	l = Hidden( 'h1', 3)
	h1 = Hidden( 'h2', 6)
	h2 = Hidden( 'h3', 6)
	p3 = Input('inputs', 9) > [
			(1, h1 > l > Output( 'o22', 3) ),
			(1, h2 > l > Output( 'o33', 3) )
		]

	# print h2.nbInputs
	# print h1.inputs
	# print h2.nbOutputs
	# print '---'
	# print p3
	print p3.edges
	print p3.layers
	p3.train()

t2()