from collections import OrderedDict
from wrappers import TheanoFunction
import Mariana.settings as MSET
import types

__all__= ["Network", "OutputMap"]

TYPE_INPUT_LAYER = "input"
TYPE_OUTPUT_LAYER = "output"
TYPE_HIDDEN_LAYER = "hidden"

def loadModel(filename) :
	"""Shorthand for Network.load"""
	return Network.load(filename)

def loadModel_old(filename) :
	"""Shorthand for Network.load_old"""
	return Network.load_old(filename)

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
		if type(outputLayer) is types.StringType :
			ol = self.network[outputLayer]
		else :
			ol = outputLayer

		self.outputFcts[ol].printGraph()

	def addOutput(self, outputLayer, fct) :
		self.outputFcts[outputLayer] = fct

	def callTheanoFct(self, outputLayer, **kwargs) :
		if type(outputLayer) is types.StringType :
			ol = self.network.layers[outputLayer]
		else :
			ol = outputLayer

		return self.outputFcts[ol](**kwargs)

	def __call__(self, outputLayer, **kwargs) :
		return self.callTheanoFct(outputLayer, **kwargs)

	def __repr__(self) :
		os = []
		for o, v in self.outputFcts.iteritems() :
			os.append(o.name)
		
		os = ', '.join(os)
		return "<theano fct '%s' for layers: '%s'>" % (self.name, os)

class Network(object) :
	"""All theano functions of all layers are accessible through the network interface **network.x(...)**."""
	def __init__(self) :
		self.inputs = OrderedDict()
		self.layers = OrderedDict()
		self.outputs = OrderedDict()
		self.layerAppelidos = {}
	
		self.edges = OrderedDict()
		
		self.outConnections = {}
		self.inConnections = {}

		self.parameters = []

		self._mustInit = True
		self.outputMaps = {}
		self.log = []

	def logEvent(self, entity, message, parameters = {}) :
		"Adds a log event to self.log. Entity can be anything hashable, Message should be a string and parameters and dict: param_name => value"
		import time, types
		assert type(message) is types.StringType
		assert type(parameters) is types.DictType
			
		entry = {
			"date": time.ctime(),
			"timestamp": time.time(),
			"message": message,
			"parameters": parameters,
			"entity": entity
		}
		self.log.append(entry)
	
	def logNetworkEvent(self, message, parameters = {}) :
		self.logEvent("Network", message, parameters)

	def logLayerEvent(self, layer, message, parameters = {}) :
		"Adds a log event to self.log. Message should be a string and parameters and dict: param_name => value"
		import time, types
		assert type(message) is types.StringType
		assert type(parameters) is types.DictType
		self.logEvent(layer.name, message, parameters)

	def printLog(self) :
		"Print a very pretty version of self.log. The log should contain all meaningful events in a chronological order"
		
		errMsg = ""
		try :
			self.init()
		except Exception as e :
			errMsg = "----OUCH----\nUnable to initialize network: %s\n------------" % e

		t = " The story of how it all began "
		t = "="*len(t) + "\n" + t + "\n" + "="*len(t)
		
		es = []
		for e in self.log :
			ps = []
			for param, value in e["parameters"].iteritems() :
				ps.append( "    -%s: %s" % (param, value) )
			ps = '\n' + '\n'.join(ps)
			# es.append("-[%s]@%s(%s), %s.\n%s" % (e["entity"], e["timestamp"], e["date"], e["message"], ps))
			es.append("-@%s(%s):\n  %s -> %s.%s" % (e["timestamp"], e["date"], e["entity"], e["message"], ps))
			
		es = '\n'.join(es)

		print "\n" + t + "\n\n" + es + "\n" + errMsg + "\n"

	def _addEdge(self, layer1Name, layer2Name) :
		"""Add a connection between two layers"""

		layer1 = self.layers[layer1Name]
		layer2 = self.layers[layer2Name]

		self.edges[ (layer1.name, layer2.name) ] = (layer1, layer2)

		try :
			self.outConnections[layer1].add(layer2)
		except :
			self.outConnections[layer1] = set([layer2])

		try :
			self.inConnections[layer2].add(layer1)
		except :
			self.inConnections[layer2] = set([layer1])

		self.logNetworkEvent("New edge %s > %s" % (layer1.name, layer2.name))

	def _addLayer(self, h) :
		"""adds a layer to the network"""
		global TYPE_INPUT_LAYER, TYPE_OUTPUT_LAYER
		
		try :
			if self.layerAppelidos[h.name] != h.appelido :
				raise ValueError("There's already a layer by the name of '%s'" % (h.name))
		except KeyError :
			self.layerAppelidos[h.name] = h.appelido
		
		self.layers[h.name] = h
		try :
			self.inConnections[h] = self.inConnections[h].union(h.network.inConnections[h])
			self.outConnections[h] = self.outConnections[h].union(h.network.outConnections[h])
		except KeyError :
			try :
				self.inConnections[h] = h.network.inConnections[h]
				self.outConnections[h] = h.network.outConnections[h]
			except KeyError :
				self.inConnections[h] = set()
				self.outConnections[h] = set()

		if h.type == TYPE_INPUT_LAYER :
			self.inputs[h.name] = h
			self.logNetworkEvent("New Input layer %s" % (h.name))
		elif h.type == TYPE_OUTPUT_LAYER :
			self.outputs[h.name] = h
			self.logNetworkEvent("New Output layer %s" % (h.name))
		else :
			self.logNetworkEvent("New Hidden layer %s" % (h.name))

	def merge(self, fromLayer, toLayer) :
		"""Merges the networks of two layers together. fromLayer must be part of the self"""
		
		self.logNetworkEvent("Merging nets: %s and %s" % (fromLayer.name, toLayer.name))

		if fromLayer.name not in self.layers :
			raise ValueError("from layer '%s' is not part of this network" % fromLayer.name)

		newLayers = toLayer.network.layers.values()
		for l in newLayers :
			self._addLayer(l)

		for e in toLayer.network.edges.iterkeys() :
			self._addEdge(e[0], e[1])

		self._addEdge(fromLayer.name, toLayer.name)
		
		for l in newLayers :
			l.network = self
		print self.layers
	def init(self) :
		"Initialiases the network by initialising every layer."

		if self._mustInit :
			self.logNetworkEvent("Initialization begins!")
			print "\n" + MSET.OMICRON_SIGNATURE

			if len(self.inputs) < 1 :
				raise ValueError("Network has no inputs")

			for inp in self.inputs.itervalues() :
				inp._init()
	
			for l in self.layers.itervalues() :
				self.parameters.extend(l.getParameters())
	
			for o in self.layers.itervalues() :
				for k, v in o.__dict__.iteritems() :
					if ( v.__class__ is TheanoFunction ) or issubclass(v.__class__, TheanoFunction) :
						if k not in self.outputMaps :
							self.outputMaps[k] = OutputMap(k, self)
						self.outputMaps[k].addOutput(o, v)
			
			self._mustInit = False

	def help(self) :
		"""prints the list of available model functions, such as train, test,..."""
		self.init()
		os = []
		for o in self.outputMaps.itervalues() :
			os.append(repr(o))
		os = '\n\t'.join(os)

		print "Available model functions:\n%s\n" % os

	@classmethod
	def isLayer(cls, obj) :
		try :
			return obj.isLayer
		except AttributeError :
			return False

	def save(self, filename) :
		import cPickle, pickle
		self.init()
		
		ext = '.mar.mdl.pkl'
		if filename.find(ext) < 0 :
			fn = filename + ext
		else :
			fn = filename

		res = {
			"edges": self.edges.keys(),
			"log": self.log,
			"layers": {}
		}
		
		for l in self.layers.itervalues() :
			sumary = {
				"class": l.__class__,
				"arguments": {
					"args": [],
					"kwargs": {}
				},
				"parameters": {},
				"needs": set()
			}

			for v in l.creationArguments["args"] :
				if self.isLayer(v) :
					if v.name not in self.layers :
						raise ValueError("Unable to save, layer '%s' is an argument to layer '%s' but is not part of the network" % (v.name, l.name))
					sumary["arguments"]["args"].append("MARLAYER.%s" % v.name)
					sumary["needs"].add(v.name)
				else :
					sumary["arguments"]["args"].append(v)

			for k, v in l.creationArguments["kwargs"].iteritems() :
				if self.isLayer(v) :
					if v.name not in self.layers :
						raise ValueError("Unable to save, layer '%s' is an argument to layer '%s' but is not part of the network" % (v.name, l.name))
					sumary["arguments"]["kwargs"][k] = "MARLAYER.%s" % v.name
					sumary["needs"].add(v.name)
				else :
					sumary["arguments"]["kwargs"][k] = v

			for k, v in l.getParameterDict().iteritems() :
				sumary["parameters"][k] = v

			res["layers"][l.name] = sumary

		f = open(fn, 'wb', pickle.HIGHEST_PROTOCOL)
		cPickle.dump(res, f)
		f.close()

	@classmethod
	def load(cls, filename) :
		"""Loads a model from disk"""
		import cPickle

		ext = '.mar.mdl.pkl'
		if filename.find(ext) < 0 :
			fn = filename + ext
		else :
			fn = filename

		f = open(fn)
		model = cPickle.load(f)

		expandedLayers = {}
		while len(expandedLayers) < len(model["layers"]) :
			for name, stuff in model["layers"].iteritems() :
				if name not in expandedLayers :
					if len(stuff["needs"]) == 0 :
						expandedLayers[name] = stuff["class"](*stuff["arguments"]["args"], **stuff["arguments"]["kwargs"])
					else :
						if len(stuff["needs"] - set(expandedLayers.keys())) == 0 :
							for i, v in enumerate(stuff["arguments"]["args"]) :
								if type(v) == types.StringType and v.find("MARLAYER") == 0 :
									stuff["arguments"]["args"][i] = expandedLayers[v.split(".")[1]]

							for k, v in stuff["arguments"]["kwargs"].iteritems() :
								if type(v) == types.StringType and v.find("MARLAYER") == 0 :
									stuff["arguments"]["kwargs"][k] = expandedLayers[v.split(".")[1]]
							
							expandedLayers[name] = stuff["class"](*stuff["arguments"]["args"], **stuff["arguments"]["kwargs"])

		for l1, l2 in model["edges"] :
			network = expandedLayers[l1] > expandedLayers[l2]
		
		return network

	@classmethod
	def load_old(cls, filename) :
		"""Loads a model from disk, saved using the ol' protocole"""
		import cPickle

		ext = '.mar.mdl.pkl'
		if filename.find(ext) < 0 :
			fn = filename + ext
		else :
			fn = filename

		f = open(fn)
		model = cPickle.load(f)

		for l1, l2 in model["edges"] :
			network = model["layers"][l1] > model["layers"][l2]
		
		return network

	def saveParameters(self, filename) :
		"""Save model parameters in HDF5 or numpy files if not succesfull"""
		try :
			self.saveParametersH5(filename)
		except :
			self.saveParametersNPY(filename)

	def saveParametersH5(self, filename) :
		"""Save model parameters in a HDF5 file with extension .mar.prm.h5 """
		try :
			import h5py
		except :
			raise ValueError("You need to install h5py to be able to save parameters")
	
		import time
		
		self.init()
		
		ext = '.mar.prm.h5'
		if filename.find(ext) < 0 :
			fn = filename + ext
		else :
			fn = filename

		f = h5py.File(fn, 'w')
		f.attrs['saveTime'] = time.time()
		f.attrs['nbLayers'] = len(self.layers)
		f.attrs['nbEdges'] = len(self.edges)
		f.attrs['nbInputs'] = len(self.inputs)
		f.attrs['nbOutputs'] = len(self.outputs)
		gLayers = f.create_group('layers')

		for l in self.layers.itervalues():
			gLayer = f.create_group(l.name)
			for pName, param in l.getParameterDict().iteritems() :
				val = param.get_value()
				dSet = gLayer.create_dataset(pName, val.shape, dtype=val.dtype)
				dSet[:] = val
		
		f.flush()
		f.close()

	def saveParametersNPY(self, filename) :
		"""Save model parameters in numpy binary files grouped into a folder"""
		import os, numpy
		
		self.init()
		
		ext = '.mar.prm.npy'
		if filename.find(ext) < 0 :
			folder = filename + ext
		else :
			folder = filename

		if not os.path.isdir(folder) :
			os.mkdir(folder)

		for l in self.layers.itervalues() :
			for pName, param in l.getParameterDict().iteritems() :
				fn = "%s_%s.npy" % (l.name, pName)
				path = os.path.join( folder, fn )
				val = param.get_value()
				numpy.save(path, val)
				
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
			g.append("\t%s -> %s" % (aidis[e[0]], aidis[e[1]]))

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
			# a bit too hacky, but solves the following: Pickle asks for attribute not found in networks which triggers initializations
			# of free outputs, and then theano complains that the layer.outputs are None, and everything crashes miserably. 
			if k == "__getstate__" or k == "__slots__" :
				raise e
			
			outs = object.__getattribute__(self, 'outputs')
			init = object.__getattribute__(self, 'init')
			init()

			maps = object.__getattribute__(self, 'outputMaps')
			try :
				return maps[k]
			except KeyError :
				raise e
	