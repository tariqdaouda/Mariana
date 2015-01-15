import numpy
import theano
import cPickle
import theano.tensor as tt

class Layer(object) :

	def __init__(self, name, inputs, nbInputs, nbOutputs, activation = None) :
		"A generic definition a layer"
		self.reset(name, inputs, nbInputs, nbOutputs, activation)

	def reset(self, name, inputs, nbInputs, nbOutputs, activation = None) :
		"resets everything to new parameters"

		self.nbInputs = nbInputs
		self.inputs = inputs
		self.nbOutputs = nbOutputs
		
		self.name = name
		self.activation = activation
		
		initWeights = numpy.random.random((nbInputs, nbOutputs)) 
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

	def serialize(self) :
		"returns a dict {name, nbInputs, nbOutputs, activation, W, b}"
		return {
				"name" : self.name,
				"nbInputs" : self.nbInputs,
				"nbOutputs" : self.nbOutputs,
				"activation" : self.activation,
				"W" : self.W.get_value(borrow = True),
				"b" : self.b.get_value(borrow = True)
			}

	def __str__ (self) :
		if self.nbOutputs < 21:
			o = " O"*self.nbOutputs
		else :
			o = " O O O ... O O O"
		return "%s: [%s ](%s x %s)" % (self.name, o, self.nbInputs, self.nbOutputs)

class NeuralNet(object) :

	def __init__(self, name, nbInputs, costFct, lr = 0.01, momentum = 0, l1 = 0., l2 = 0.) :
		"A neural network"
		self.reset(name, nbInputs, costFct, lr, momentum, l1, l2)

	def reset(self, name, nbInputs, costFct, lr, momentum, l1, l2) :
		self.name = name
		self.nbInputs = nbInputs
		self.costFct = costFct
		
		self.inputs = tt.matrix(name = self.name + "_X")
		self.y = tt.ivector(name = self.name + "_Y")
		self.layers = []
		self.layersDct = {}
		self.params = []
		self.lr = lr
		self.momentum = momentum
		self.l1 = l1
		self.l2 = l2
		
		self._mustInitUpdates = True

	def stackLayer(self, name, nbOutputs, activation) :
		"adds a layer to the stack and returns it"
		if name in self.layersDct :
			raise KeyError("There's already a layer by the name '%s'" % name)

		if len(self.layers) < 1 :
			layer = Layer(self.name + "_" + name, self.inputs, self.nbInputs, nbOutputs, activation)
		else :
			priorLayer = self.layers[-1]
			layer = Layer(self.name + "_" + name, priorLayer.outputs, priorLayer.nbOutputs, nbOutputs, activation)

		self.layersDct[name] = (layer, len(self.layers))
		self.layers.append(layer)
		return layer

	def popLayer(self) :
		"removes the last layer from the stack and returns it"
		layer = self.layers.pop()
		del(self.layersDct[layer.name])

		self._mustInitUpdates = True

		return layer

	def _initUpdates(self) :
		self.outputs = self.layers[-1].outputs
		# cost = self.costFct(self.y, self.outputs)
		self.updates = []
		for layer in self.layers :
			self.params.extend(layer.params)
			for param in layer.params :
				# gparam = tt.grad(cost, param) + self.momentun * ()
				gparam = tt.grad(cost, param)
				momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				self.updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
				self.updates.append((param, param - self.lr * momentum_param))
				
				
		L1 =  self.l1 * sum([abs(l.W).sum() for l in self.layers])
		L2 = self.l2 * sum([(l.W**2).sum() for l in self.layers])
		cost = self.costFct(self.y, self.outputs) + L1 + L2

		self.theano_train = theano.function(inputs = [self.inputs, self.y], outputs = [cost, self.outputs], updates = self.updates)
		self.theano_test = theano.function(inputs = [self.inputs, self.y], outputs = [cost, self.outputs])
		self.theano_propagate = theano.function(inputs = [self.inputs], outputs = self.outputs)
		self.theano_prediction = theano.function(inputs = [self.inputs], outputs = tt.argmax(self.outputs, axis = 1))
		
		self._mustInitUpdates = False

	def train(self, x, y) :
		if self._mustInitUpdates :
			self._initUpdates()
		# print x.shape
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
		return self.theano_prediction(x)

	def save(self, filename) :
		"save the whole model"
		fil = open(filename + '.mdl', 'wb')
		
		model = {
			"name" : self.name,
			"nbInputs" : self.nbInputs,
			"costFct" : self.costFct,
			"lr" : self.lr,
			"momentum" : self.momentum,
			"l1" : self.l1,
			"l2" : self.l2,
			"layers" : []
		}

		layers = []
		for layer in self.layers :
			params = []
			layers.append(layer.serialize())

		model["layers"] = layers
		cPickle.dump(model, fil, -1)
		fil.close()

	@classmethod
	def load(cls, filename) :
		"load a previously saved model"
		fil = open(filename)
		model = cPickle.load(fil)
		nn = NeuralNet( model["name"], model["nbInputs"], model["costFct"], model["lr"], model["momentum"], model["l1"], model["l2"])
		for layer in model.layers :
			l = nn.stackLayer(layer["name"], layer["nbOutputs"], layer["activation"])
			l.W = layer["W"]
			l.b = layer["b"]
		fil.close()
		return nn

	def __getitem__(self, layerName) :
		return self.layersDct[layerName]
	
	def __str__(self) :
		ls = []
		s = "<Net: %s (inputs: %s, cost: %s, lr: %s, momentum: %s, l1: %s, l2: %s)>" % (self.name, self.nbInputs, self.costFct.__name__, self.lr, self.momentum, self.l1, self.l2)
		for l in self.layers :
			strl = str(l)
			ls.append( ' '* (len(s)/2) + 'X' )
			ls.append(' '*( (len(s)-len(strl))/2) + strl)
			prevLen = strl
			
		s += "\n\n%s" % ('\n'.join(ls))
		return s