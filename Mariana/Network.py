from collections import OrderedDict
from wrappers import TheanoFunction

class OutputMap(object):
	"""
	Encapsulates outputs as well as their theano functions.
	The role of an output map object is to applies a function such as theano_train to the set outputs that have it
	"""
	def __init__(self, name):
		self.name = name
		self.outputFcts = {}

	def addOutput(self, output, fct) :
		self.outputFcts[output.name] = fct

	def map(self, outputLayerName, *args) :
		return self.outputFcts[outputLayerName](*args)

	def __call__(self, outputLayerName, *args) :
		return self.map(outputLayerName, *args)

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
	def __init__(self) :
		self.inputs = OrderedDict()
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
		self.edges.add( (layer1, layer2))

	def addInput(self, i) :
		"""adds an input to the layer"""
		self.inputs[i.name] = i

	def addOutput(self, o) :
		"""adds an output o to the network"""
		self.outputs[o.name] = o

	def merge(self, fromLayer, toLayer) :
		"""Merges two layer together. There can be only one input to a network, if self ans network both have an input layer
		this function will raise a ValueError."""
		# if fromLayer.name not in self.layers :
		# 	raise ValueError("from layer '%s' is not part of this network" % fromLayer.name)

		for inp in toLayer.network.inputs.itervalues() :
			self.addInput(inp)
	
		self.addEdge(fromLayer, toLayer)
		
		for o in toLayer.network.outputs.itervalues() :
			self.addOutput(o)
	
		self.layers.update(toLayer.network.layers)
		self.edges = self.edges.union(toLayer.network.edges)

	def init(self) :
		"Initialiases the network by initialising every layer."
		if self._mustInit :
			for inp in self.inputs.itervalues() :
				inp._init()

			self._mustInit = False

			for o in self.outputs.itervalues() :
				for k, v in o.__dict__.iteritems() :
					if ( v.__class__ is TheanoFunction ) or issubclass(v.__class__, TheanoFunction) :
						if k not in self.outputMaps :
							self.outputMaps[k] = OutputMap(k)
						self.outputMaps[k].addOutput(o, v)
			# print self.outputMaps, self.outputs.values()

	def __getattribute__(self, k) :
		"""All theano_x functions are accessible through the network interface network.x(). Here x is called a model function"""
		try :
			return object.__getattribute__(self, k)
		except AttributeError as e :
			maps = object.__getattribute__(self, 'outputMaps')
			init = object.__getattribute__(self, 'init')
			init()
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

	def toDOT(self, name, forceInit = True) :
		"""returns a string representing the network in the DOT language.
		If forceInit, the network will first try to initialize each layer
		before constructing the graph"""

		import time

		if forceInit :
			self.init()

		com = "//Mariana network DOT representation generated on %s" % time.ctime()
		s = "#COM#\ndigraph %s{\n#HEAD#;\n\n#GRAPH#;\n}" % name
		
		headers = []
		for l in self.layers.itervalues() :
			headers.append("\t" + l._dot_representation())

		g = []
		for e in self.edges :
			g.append("\t%s -> %s" % (e[0].name, e[1].name))
	
		s = s.replace("#COM#", com)
		s = s.replace("#HEAD#", ';\n'.join(headers))
		s = s.replace("#GRAPH#", ';\n'.join(g))

		return s

	def saveDOT(self, name, forceInit = True) :
		"saves the current network as a graph in the DOT format into the file name.mariana.dot"
		f = open(name + '.mariana.dot', 'wb')
		f.write(self.toDOT(name, forceInit))
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