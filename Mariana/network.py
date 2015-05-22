from collections import OrderedDict
from wrappers import TheanoFunction

class OutputMap(object):
	"""
	Encapsulates outputs as well as their theano functions.
	The role of an output map object is to apply a function such as theano_train to the set of outputs that have it
	"""
	def __init__(self, name):
		self.name = name
		self.outputFcts = {}

	def printGraph(self, outputName) :
		"""Print the theano graph of the function associated with a given output"""
		self.outputFcts[outputName].printGraph()
	
	def addOutput(self, output, fct) :
		self.outputFcts[output.name] = fct

	def map(self, outputLayerName, **kwargs) :
		return self.outputFcts[outputLayerName](**kwargs)

	def __call__(self, outputLayerName, **kwargs) :
		return self.map(outputLayerName, **kwargs)

	def __repr__(self) :
		os = []
		for o, v in self.outputFcts.iteritems() :
			os.append(o)
		os = ', '.join(os)
		return "<'%s' for outputs: %s>" % (self.name, os)

class Network(object) :
	"""All theano_x functions of the outputs are accessible through the network interface network.x().
	Here x is called a model function. Ex: if an output layer has a theano function named theano_classify(),
	calling net.classify(), will apply out.theano_classify(). The result will be a dictionary of { "output name" => result of the theano function}"""	
	def __init__(self) :
		self.inputs = OrderedDict()
		self.layers = OrderedDict()
		self.outputs = OrderedDict()
		self.layerConnectionCount = {}

		self.regularizations = OrderedDict()
		self.edges = set()

		self.params = []

		self._mustInit = True
		self.outputMaps = {}

	def addParams(self, params) :
		"""Add parameters to the network"""
		self.params.extend(params)

	def addEdge(self, layer1, layer2) :
		"""Add a connection between two layers"""
		self.layers[layer1.name] = layer1
		self.layers[layer2.name] = layer2
		self.edges.add( (layer1, layer2))
		try :
			self.layerConnectionCount[layer1.name] += 1
		except KeyError
			self.layerConnectionCount[layer1.name] = 1

		try :
			self.layerConnectionCount[layer2.name] += 1
		except KeyError
			self.layerConnectionCount[layer2.name] = 1

	def removeEdge(self, layer1, leyer2) :
		"""Remove the connection between two layers"""
		def _del(self, layer) :
			ds = [self.inputs, self.outputs, self.layers]
			if self.layerConnectionCount[layer.name] < 1 :
				for d in ds :
					if layer.name in d :
						del(d[layer.name])

		self.edges.remove( (layer1, layer2))
		self.layerConnectionCount[layer1] -= 1
		self.layerConnectionCount[layer2] -= 1

		_del(self, layer1)
		_del(self, layer2)

	def addInput(self, i) :
		"""adds an input to the layer"""
		self.inputs[i.name] = i

	def addOutput(self, o) :
		"""adds an output o to the network"""
		self.outputs[o.name] = o

	def merge(self, fromLayer, toLayer) :
		"""Merges two layer together. There can be only one input to a network, if self ans network both have an input layer
		this function will raise a ValueError."""
		if fromLayer.name not in self.layers :
			raise ValueError("from layer '%s' is not part of this network" % fromLayer.name)

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
		s = '#COM#\ndigraph "%s"{\n#HEAD#;\n\n#GRAPH#;\n}' % name
		
		headers = []
		aidi = 0
		aidis = {}
		for l in self.layers.itervalues() :
			aidis[l.name] = "layer%s" % aidi
			headers.append("\t" + aidis[l.name] + l._dot_representation())
			aidi += 1

		g = []
		for e in self.edges :
			g.append("\t%s -> %s" % (aidis[e[0].name], aidis[e[1].name]))
	
		s = s.replace("#COM#", com)
		s = s.replace("#HEAD#", ';\n'.join(headers))
		s = s.replace("#GRAPH#", ';\n'.join(g))
		s = s.replace("-", '_')

		return s

	def saveDOT(self, name, forceInit = True) :
		"saves the current network as a graph in the DOT format into the file name.mariana.dot"
		f = open(name + '.mariana.dot', 'wb')
		f.write(self.toDOT(name, forceInit))
		f.close()

	def __repr__(self) :
		return "<Net (%s layers): %s > ... > [%s]>" % (len(self.layers), self.inputs.keys(), self.outputs.keys())

	def __getattribute__(self, k) :
		"""All theano functions are accessible through the network interface network.x(). Here x is called a model function"""
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