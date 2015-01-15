from abc import ABCMeta

import numpy
import theano
import theano.tensor as tt

# net = I > H1 > H2 > O
#
# I > H1 > (0.3. H3) > O
# I > H2 > (0.7, H3) > O
# net = I.getNet()
#
# net = I > [H1, H2] > {'layer': H3, 'H2' : 0.2, 'H1' : 0.8} > O

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

	def feedsTo(self, layer) :
		"""Connects two layers"""
		if layer.name in self.feedsInto :
			raise KeyError("Already feeding to a layer by the name of '%s'" % layer.name)

		if len(self.feedsInto) > 1 :
			raise NotImplemented("For now Mariana allows only one input for each layer")

		if self.nbInputs is None :
			self.nbInputs = layer.nbOutputs
		elif self.nbInputs != layer.nbOutputs :
			raise ValueError("'%s' has a number of Outputs of '%s', while the previous one had '%s'" % (layer.name, layer.nbOutputs, self.nbInputs))

		self.feedsInto[layer.name] = layer	
		layer.feededBy[self.name] = self

		return self

	def _setInputs(self, layer) :
		self.nbInputs = layer.nbOutputs

	def disconnect(self, layer) :
		"disconnect 'layer' from self"
		try :
			del(self.feedsInto[layer.name])
		except KeyError :
			raise KeyError("This layer does not feed to a layer named '%s'" % layer.name)
	
	# @abstractmethod
	def _init(self) :
		pass

	def __gt__(self, layer) :
		"""Layers are connected using the '>' operator: layer1 > layer2"""
		self.feedsTo(layer)

	def __ne__(self, layer) :
		"""Layers are disconnected using the '!=' operator: layer1 != layer2"""
		self.disconnect(layer)

	def __str__(self) :
		return "Layer '%s' (%s units) " % (self.name, self.nbInputs)

	def __repr__(self) :
		return str(self)

	def printNetworkFromHere(self, indent = 0) :
		s = [ "%s>%s" % ( "--" * indent, str(self) ) ]
		for l in self.feedsInto.itervalues() :
			s.append(l.printNetworkFromHere(indent + 1))
		return "\n".join(s)

class Network(object) :

	def __init__(self, inputLayer) :
		self.inputLayer = inputLayer
		

class Input(LayerABC) :

	def __init__(self, name, nbInputs) :
		LayerABC.__init__(self, name)
		self.nbInputs = nbInputs
		self.inputs = tt.matrix(name = self.name + "_X")

	def feedsTo(self, layer) :
		LayerABC.feedsTo(layer)
		return Network(self)

	def _init(self) :
		self.outputs = self.inputs

class Output(LayerABC) :

	def __init__(self, name, costFct, lr = 0.01, momentum = 0, l1 = 0., l2 = 0.) :
		LayerABC.__init__(self, name)
		
	def _init(self) :
		self.outputs = self.inputs
		L1 =  self.l1 * sum([abs(l.W).sum() for l in self.layers])
		L2 = self.l2 * sum([(l.W**2).sum() for l in self.layers])
		cost = self.costFct(self.y, self.outputs) + L1 + L2

class Hidden(LayerABC) :

	def __init__(self, name, activation = None) :
		LayerABC.__init__(self, name)
		self.activation = activation
		self.inputs = None

	def _init(self) :
		prev = self.feededBy.values[0]
		self.inputs = prev.outputs

		initWeights = numpy.random.random((self.nbInputs, self.nbOutputs)) 
		initWeights = (initWeights/sum(initWeights))
		initWeights = numpy.asarray(initWeights, dtype=theano.config.floatX)
		self.W = theano.shared(value = initWeights, name = self.name + "_W")
			
		initBias = numpy.zeros((nbOutputs,), dtype=theano.config.floatX)
		self.b = theano.shared(value = initBias, name = self.name + "_b")

		self.params = [self.W, self.b]
		
		if self.activation is None :
			self.outputs = tt.dot(self.inputs, self.W) + self.b
		else :
			self.outputs = self.activation(tt.dot(self.inputs, self.W) + self.b)

# net = Inputs('in', 33) > Hidden(5) > Output(2)
# net.train()
# print a