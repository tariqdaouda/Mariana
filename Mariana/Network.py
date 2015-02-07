from collections import OrderedDict

class OutputMap(object):
	"""Applies a function such as theano_train to a set outputs"""
	def __init__(self, name):
		self.name = name
		self.outputFcts = []

	def addOutput(self, output, fct) :
		self.outputFcts.append((output, fct))

	def map(self, *args) :
		res = {}
		for o, v in self.outputFcts :
			res[o.name] = v(*args)
		return res

	def __call__(self, *args) :
		return self.map(*args)

	def __repr__(self) :
		os = []
		for o, v in self.outputFcts :
			os.append(o.name)
		os = ', '.join(os)
		return "<%s for output: %s>" % (self.name, os)

class Network(object) :

	def __init__(self, entryLayer, inputLayer = None) :
		self.entryLayer = entryLayer
		self.inputLayer = inputLayer
		self.layers = OrderedDict()
		self.outputs = OrderedDict()
		self.edges = set()

		self.params = []

		self._mustInit = True
		self.outputMaps = {}

	def addParams(self, params) :
		self.params.extend(params)

	def addEdge(self, layer1, layer2) :
		self.layers[layer1.name] = layer1
		self.layers[layer2.name] = layer2
		self.edges.add( (layer1.name, layer2.name))

	def addOutput(self, o) :
		"""adds an output o to the network"""
		self.outputs[o.name] = o

	def merge(self, conLayer, network) :
		if network.inputLayer is not None and network.inputLayer is not self.inputLayer :
			raise ValueError("Can't merge, the network already has an input layer")

		self.addEdge(conLayer, network.entryLayer)
		
		network.entryLayer = self.entryLayer
		for o in network,outputs :
			self.addOutput(o)
		self.layers.update(network.layers)
		self.edges = self.edges.union(network.edges)

	def _init(self) :
		if self._mustInit :
			self.entryLayer._init()
			self._mustInit = False

			for o in self.outputs.itervalues() :
				for k, v in o.__dict__.iteritems() :
					if k.find("theano") == 0 :
						name = k.replace("theano_", "")# + "_model"
						if name not in self.outputMaps :
							self.outputMaps[name] = OutputMap(name)
						self.outputMaps[name].addOutput(o, v)
			print self.outputMaps

	def __getattribute__(self, k) :
		try :
			return object.__getattribute__(self, k)
		except AttributeError as e :
			maps = object.__getattribute__(self, 'outputMaps')
			_init = object.__getattribute__(self, '_init')
			_init()
			try :
				return maps[k]
			except KeyError :
				raise e

	def help(self) :
		"""prints the list of available model functions, such as train, test,..."""
		os = []
		for o in self.outputMaps.itervalues() :
			os.append(repr(o))
		os = '\n\t'.join(os)
		
		print "Available model functions:\n" % os

	# def train(self, x, y) :
	# 	self._init()
	# 	ret = {}
	# 	for o in self.outputs.itervalues() :
	# 		ret[o.name] = o.theano_train(x, y)
	# 	return ret

	# def test(self, x, y) :
	# 	self._init()
	# 	ret = {}
	# 	for o in self.outputs.itervalues() :
	# 		ret[o.name] = o.theano_test(x, y)
	# 	return ret

	# def propagate(self, x) :
	# 	self._init()
	# 	ret = {}
	# 	for o in self.outputs.itervalues() :
	# 		ret[o.name] = o.theano_propagate(x)
	# 	return ret

	# def predict(self, x) :
	# 	self._init()
	# 	ret = {}
	# 	for o in self.outputs.itervalues() :
	# 		ret[o.name] = o.theano_predict(x)
	# 	return ret

	def save(self, filename) :
		"save the model into filename.mariana.pkl"
		import cPickle
		f = open(filename + '.mariana.pkl', 'wb')
		cPickle.dump(self, f, -1)
		f.close()

	def __repr__(self) :
		s = []
		for o in self.outputs :
			s.append(o)

		if self.inputLayer is None :
			inp = None
		else :
			inp = self.inputLayer.name

		return "<Net (%s layers): %s > ... > [%s]>" % (len(self.layers), inp, ', '.join(s))