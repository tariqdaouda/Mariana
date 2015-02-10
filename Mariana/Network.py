from collections import OrderedDict

class OutputMap(object):
	"""
	Encapsulates outputs as well as their theano functions.
	The role of an output map object is to applies a function such as theano_train to the set outputs that have it
	"""
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
	"""All theano_x functions of the outputs are accessible through the network interface network.x().
	Here x is called a model function. Ex: if an output layer has a theano function named theano_classify(),
	calling net.classify(), will apply out.theano_classify(). The result will be a dictionary of { "output name" => result of the theano function}"""	
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
		"""Add parameters to the network"""
		self.params.extend(params)

	def addEdge(self, layer1, layer2) :
		"""Add a connection between two layers to the network"""
		self.layers[layer1.name] = layer1
		self.layers[layer2.name] = layer2
		self.edges.add( (layer1.name, layer2.name))

	def addOutput(self, o) :
		"""adds an output o to the network"""
		self.outputs[o.name] = o

	def merge(self, conLayer, network) :
		"""Merges two layer together. There can be only one input to a network, if self ans network both have an input layer
		this function will raise a ValueError."""
		if network.inputLayer is not None and network.inputLayer is not self.inputLayer :
			raise ValueError("Can't merge, the network already has an input layer")

		self.addEdge(conLayer, network.entryLayer)
		
		network.entryLayer = self.entryLayer
		for o in network,outputs :
			self.addOutput(o)
		self.layers.update(network.layers)
		self.edges = self.edges.union(network.edges)

	def _init(self) :
		"Initialiases the network by initialising every layer"
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
			# print self.outputMaps, self.outputs.values()

	def __getattribute__(self, k) :
		"""All theano_x functions are accessible through the network interface network.x(). Here x is called a model function"""
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