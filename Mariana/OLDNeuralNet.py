import numpy
import theano
import theano.tensor as tt

class Layer(object) :

	def __init__(self, name, inputs, nbInputs, nbOutputs, activation = None) :
		
		self.nbInputs = nbInputs
		self.nbOutputs = nbOutputs
		
		self.inputs = inputs
		self.name = name
		self.activation = activation
		
		#simply init with small random weights, should use glorot et al. 2010
		tmpW = numpy.random.random((nbInputs, nbOutputs)) 
		tmpW = (tmpW/sum(tmpW)) #*0.001
		initW = numpy.asarray(tmpW, dtype=theano.config.floatX)
		self.W = theano.shared(value = initW, name = self.name + "_W")
		
		initB = numpy.zeros((nbOutputs,), dtype=theano.config.floatX)
		self.b = theano.shared(value = initB, name = self.name + "_b")
		
		self.params = [self.W, self.b]
		
		if self.activation is None :
			self.outputs = tt.dot(self.inputs, self.W) + self.b
		else :
			self.outputs = self.activation(tt.dot(self.inputs, self.W) + self.b)

	def __str__ (self) :
		return "Layer: %s" % self.name

class NeuralNet(object) :

	def __init__(self, name, nbInputs, costFct, lr = 0.01, l1 = 0., l2 = 0.) :
		self.name = name
		self.nbInputs = nbInputs
		self.costFct = costFct
		self.inputs = tt.lmatrix(name = self.name + "_X")
		self.y = tt.lvector(name = self.name + "_Y")
		self.layers = []
		self.layersDct = {}
		self.params = []
		self.lr = lr
		self.l1 = l1
		self.l2 = l2
		
		self._mustInitUpdates = True

	def stackLayer(self, name, nbOutputs, activation) :
		if name in self.layersDct :
			raise KeyError("There's already a layer by the name %s" % name)

		if len(self.layers) < 1 :
			layer = Layer(self.name + "_" + name, self.inputs, self.nbInputs, nbOutputs, activation)
		else :
			priorLayer = self.layers[-1]
			layer = Layer(self.name + "_" + name, priorLayer.outputs, priorLayer.nbOutputs, nbOutputs, activation)

		self.layersDct[name] = layer
		self.layers.append(layer)
		self.params.extend(layer.params)
	
	def _initUpdates(self) :
		self.outputs = self.layers[-1].outputs
		self.cost = self.costFct(self.y, self.outputs)

		self.updates = []
		self.gs = []
		for param in self.params :
			gparam = tt.grad(self.cost, param)
			self.updates.append((param, param - self.lr * gparam))
			self.gs.append(gparam)
			
		self.theano_train = theano.function(inputs = [self.inputs, self.y], outputs = [self.cost, self.outputs], updates = self.updates)
		self.theano_test = theano.function(inputs = [self.inputs, self.y], outputs = [self.cost, self.outputs])
		self.theano_propagate = theano.function(inputs = [self.inputs], outputs = self.outputs)
		#~ self.theano_predict = theano.function(inputs = [self.inputs], outputs = tt.argmax(self.outputs))
		
		self._mustInitUpdates = False

	def train(self, x, y) :
		if self._mustInitUpdates :
			self._initUpdates()
		
		return self.theano_train(x, y)

	def test(self, x, y) :
		"same function for both test and validation"
		if self._mustInitUpdates :
			self._initUpdates()
		return self.theano_test(x, y)

	def propagate(self, x) :
		if self._mustInitUpdates :
			self._initUpdates()
		return self.theano_propagate(x)

	def predict(self, x) :
		if self._mustInitUpdates :
			self._initUpdates()
		return tt.argmax(self.theano_predict(x))

	def L1(self) :
		return sum([abs(l.W).sum() for l in self.layers])

	def L2(self) :
		return sum([(l.W**2).sum() for l in self.layers])

	def __getitem__(self, layerName) :
		return self.layersDct[layerName]
	
	def __str__(self) :
		ls = []
		for l in self.layers :
			ls.append(l.name)
		ls = " -> ".join(ls)
		s = "Net: %s (%s)" % (self.name, ls)
		return s

def negLogLikelihood(y, outputs) :
	"""cost fct for softmax"""
	cost = -tt.mean(tt.log(outputs)[tt.arange(y.shape[0]), y])
	return cost

def meanSquaredError(y, outputs) :
	"""cost fct"""
	cost = -tt.mean( tt.dot(outputs, y) **2 )
	#~ cost = -tt.mean(outputs - y)
	return cost
