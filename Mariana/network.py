from collections import OrderedDict
from wrappers import TheanoFunction
import Mariana.settings as MSET
import types

__all__= ["Network", "OutputMap"]

class OutputMap(object):
	"""
	Encapsulates outputs as well as their theano functions.
	The role of an output map object is to apply a function such as theano_train to the set of outputs it belongs to
	"""
	def __init__(self, name, network):
		self.name = name
		self.network = network
		self.outputFcts = {}

	def printGraph(self, outputLayer) :
		"""Print the theano graph of the function associated with a given output"""
		if type(outputLayer) is StringType :
			ol = self.network[outputLayer]
		else :
			ol = outputLayer

		self.outputFcts[ol].printGraph()

	def addOutput(self, outputLayer, fct) :
		self.outputFcts[outputLayer] = fct

	def callTheanoFct(self, outputLayer, **kwargs) :
		if type(outputLayer) is types.StringType :
			ol = self.network.outputs[outputLayer]
		else :
			ol = outputLayer

		return self.outputFcts[ol](**kwargs)

	def __call__(self, outputLayer, **kwargs) :
		return self.callTheanoFct(outputLayer, **kwargs)

	def __repr__(self) :
		os = []
		for o, v in self.outputFcts.iteritems() :
			os.append(o)
		os = ', '.join(os)
		return "<theano fct '%s' for output layer: '%s'>" % (self.name, os)

class Network(object) :
	"""All **theano_x** functions of the outputs are accessible through the network interface **network.x(...)**."""
	def __init__(self) :
		self.inputs = OrderedDict()
		self.layers = OrderedDict()
		self.outputs = OrderedDict()
		self.layerAppelidos = {}
		self.layerConnectionCount = {}

		self.regularizations = OrderedDict()
		self.edges = set()

		self.params = []

		self._mustInit = True
		self.outputMaps = {}

	def addEdge(self, layer1, layer2) :
		"""Add a connection between two layers"""
		for layer in (layer1, layer2) :
			try :
				if self.layerAppelidos[layer.name] != layer.appelido :
					raise ValueError("There's already a layer by the name of '%s'" % (layer.name))
			except KeyError :
				self.layerAppelidos[layer.name] = layer.appelido

		self.layers[layer1.name] = layer1
		self.layers[layer2.name] = layer2
		self.edges.add( (layer1, layer2))
		try :
			self.layerConnectionCount[layer1.name] += 1
		except KeyError :
			self.layerConnectionCount[layer1.name] = 1

		try :
			self.layerConnectionCount[layer2.name] += 1
		except KeyError :
			self.layerConnectionCount[layer2.name] = 1

	def removeEdge(self, layer1, layer2) :
		"""Remove the connection between two layers"""
		def _del(self, layer) :
			ds = [self.inputs, self.outputs, self.layers]
			if self.layerConnectionCount[layer.name] < 1 :
				for d in ds :
					if layer.name in d :
						del(d[layer.name])

		self.edges.remove( (layer1, layer2))
		self.layerConnectionCount[layer1.name] -= 1
		self.layerConnectionCount[layer2.name] -= 1

		_del(self, layer1)
		_del(self, layer2)

	def addInput(self, i) :
		"""adds an input to the layer"""
		self.inputs[i.name] = i

	def addOutput(self, o) :
		"""adds an output o to the network"""
		self.outputs[o.name] = o

	def merge(self, fromLayer, toLayer) :
		"""Merges the networks of two layers together."""
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
			print "\n" + MSET.OMICRON_SIGNATURE

			for inp in self.inputs.itervalues() :
				inp._init()

			self._mustInit = False
	
			for l in self.layers.itervalues() :
				self.params.extend(l.getParameters())
			
			for o in self.outputs.itervalues() :
				o._setTheanoFunctions()
				for k, v in o.__dict__.iteritems() :
					if ( v.__class__ is TheanoFunction ) or issubclass(v.__class__, TheanoFunction) :
						if k not in self.outputMaps :
							self.outputMaps[k] = OutputMap(k, self)
						self.outputMaps[k].addOutput(o, v)

	def help(self) :
		"""prints the list of available model functions, such as train, test,..."""
		os = []
		for o in self.outputMaps.itervalues() :
			os.append(repr(o))
		os = '\n\t'.join(os)

		print "Available model functions:\n" % os

	def savePickle(self, filename) :
		"save the model into a python pickle filename.mariana.pkl"
		import cPickle
		f = open(filename + '.mariana.pkl', 'wb')
		cPickle.dump(self, f, -1)
		f.close()

	def save(self, filename) :
		"save the model into a folder filename.mariana.model"
		import numpy, tempfile, os, shutils, cPickle
		tmpDir = tempfile.mkdtemp()

		for l in self.layers :
			for pName, param in l.getParameterDict().iteritems() :
				fn = "%s-%s.npy" % (l.name, pName)
				path = os.path.join( tmpDir, fn )
				numpy.save(path, param)
				l.initializations = []
				getattr(l, pName).set_value(None)

		path = os.path.join( tmpDir, "network.pkl" )
		f = open(path, 'wb')
		cPickle.dump(self, f, -1)
		f.close()
		shutil.move(tmpDir, filename + ".mariana.model")

	@classmethod
	def loadModel(self, folder) :
		"""loads a model from a folder"""
		import numpy, tempfile, os, shutils, cPickle

		f = open(os.path.join(folder, "network.pkl"))
		network = cPickle(f)
		f.close()
	
		for l in network.layers :
			for pName, param in l.getParameterDict().iteritems() :
				fn = "%s-%s.npy" % (l.name, pName)
				path = os.path.join( folder, fn )
				val = numpy.load(path)
				getattr(l, pName).set_value(val)

		return network

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

	def saveHTML(self, name, forceInit = True) :
		"""Creates an HTML file with the graph representation. Heavily inspired from: http://stackoverflow.com/questions/22595493/reading-dot-files-in-javascript-d3"""
		from Mariana.HTML_Templates.aqua import getHTML
		import time
		temp = getHTML(self.toDOT(name, forceInit), name, time.ctime())
		f = open(name + '.mariana.dot.html', 'wb')
		f.write(temp)
		f.close()

	def saveDOT(self, name, forceInit = True) :
		"saves the current network as a graph in the DOT format into the file name.mariana.dot"
		f = open(name + '.mariana.dot', 'wb')
		f.write(self.toDOT(name, forceInit))
		f.close()

	def __getitem__(self, l) :
		"""get a layer by name"""
		return self.layers[l]

	def __repr__(self) :
		return "<Net (%s layers): %s > ... > [%s]>" % (len(self.layers), self.inputs.keys(), self.outputs.keys())

	def __getattribute__(self, k) :
		"""
		All theano functions are accessible through the network interface network.x(). Here x is called a model function.
		"""
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
