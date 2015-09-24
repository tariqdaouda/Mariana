import Mariana.layers as ML
import numpy, random, theano

__all__ = ["DatasetHandle", "Dataset_ABC", "Series", "RandomSeries", "ClassSets", "DatasetMapper"]

class DatasetHandle(object) :
	"""Handles represent a couple (dataset, subset). They are returned by
	Datasets and are mainly intended to be used as arguments to a DatasetMapper
	to associate layers to data."""

	def __init__(self, dataset, subset) :
		self.dataset = dataset
		self.subset = subset

	def __repr__(self) :
		return "<Handle %s.%s>" % (self.dataset.__class__.__name__, self.subset)

class Dataset_ABC(object) :
	"""An abstract class representing a datatset.
	A Dataset typically contains and synchronizes and manages several subsets.
	Datasets are mainly intended to be used with DatasetMappers.
	It should take subsets as constructor aguments and return handles through
	the same interface as attributes (with a '.subset').
	"""

	def get(self, subset, i, n) :
		"""Returns n element from a subset, starting from position i"""
		raise NotImplemented("Should be implemented in child")

	def getAll(self, subset) :
		"""return all the elements from a subset"""
		raise NotImplemented("Should be implemented in child")

	def reroll(self) :
		"""re-initiliases all subset, this is where shuffles are implemented"""
		raise NotImplemented("Should be implemented in child")

	def getHandle(self, subset) :
		"""returns a DatasetHandle(self, subset)"""
		raise NotImplemented("Should be implemented in child")

	def __len__(self) :
		raise NotImplemented("Should be implemented in child")

	def __getattr__(self, k) :
		"""If no attribute by the name of k is found, look for a
		subset by that name, and if found return a handle"""

		g = object.__getattribute__(self, "getHandle")
		res = g(k)
		if res :
			return res
		raise AttributeError("No attribute or dataset by the name of '%s'" % k)

class Series(Dataset_ABC) :
	"""Synchronizes and returns the elements of several lists sequentially.
	All lists must have the same length. It expects arguments in the following form::
			
			trainSet = Series(images = train_set[0], classes = train_set[1])"""
	
	def __init__(self, **kwargs) :

		self.lists = {}
		self.length = 0
		for k, v in kwargs.iteritems() :
			if self.length == 0 :
				self.length = len(v)
			else :
				if len(v) != self.length :
					raise ValueError("All lists must have the same length, previous list had a length of '%s', list '%s' has a length of '%s" % (self.length, k, len(v)))
			self.lists[k] = numpy.asarray(v)

	def reroll(self) :
		"""Does nothing"""
		pass

	def get(self, subset, i, n) :
		"""Returns n element from a subset, starting from position i"""
		return self.lists[subset][i:i + n]

	def getAll(self, subset) :
		"""return all the elements from a subset"""
		return self.lists[subset]

	def getHandle(self, subset) :
		"""returns a DatasetHandle(self, subset)"""
		if subset in self.lists :
			return DatasetHandle(self, subset)
		return None

	def __len__(self) :
		return self.length

	def __repr__(self) :
		return "<%s len: %s, nbLists: %s>" % (self.__class__.__name__, self.length, len(self.lists))

class RandomSeries(Series) :
	"""This is very much like Series, but examples are randomly sampled with replacement"""
	def __init__(self, **kwargs) :
		"""
		Expects arguments in the following form::
		
				trainSet = RandomSeries(images = train_set[0], classes = train_set[1])"""

		Series.__init__(self, **kwargs)
		self.reroll()

	def reroll(self) :
		"""shuffles subsets but keep them synched"""
		indexes = numpy.random.randint(0, self.length, self.length)
		for k in self.lists :
			self.lists[k] = self.lists[k][indexes]

class ClassSets(Dataset_ABC) :
	"""Pass it one set for each class and it will take care of randomly sampling them with replacement.
	All class sets must have at least two elements. The subsets exposed by a ClassSets are:
		
		* .input, for the raw inputs
		* .classNumber, each class is represent by an int
		* .onehot, onehot representation of classes

		"""
	def __init__(self, **kwargs) :
		"""
		Expects arguments in the following form::
		
				trainSet = ClassSets(cars = train_set[0], bikes = train_set[1])
		"""

		self.classSets = {}
		
		self.minLength = float('inf')
		self.maxLength = 0
		self.totalLength = 0
		self.inputSize = 0
		for k, v in kwargs.iteritems() :
			self.classSets[k] = numpy.asarray(v)
			if len(self.classSets[k].shape) < 2 :
			 	self.classSets[k] = self.classSets[k].reshape(self.classSets[k].shape[0], 1)

			try :
				inputSize = len(v[0])
			except :
				inputSize = 1

			if self.inputSize == 0 :
				self.inputSize = inputSize
			elif self.inputSize != inputSize :
				raise ValueError("All class elements must have the same size. Got '%s', while the previous value was '%s'" % ( len(v[0]), self.inputSize ))
				
		if self.minLength < 2 :
			raise ValueError('All class sets must have at least two elements')

		self.classNumbers = {}
		self.onehots = {}
		self.nbElements = {}
		i = 0
		for k, v in self.classSets.iteritems() :
			length = len(v)
			if length < self.minLength :
				self.minLength = length
			if self.maxLength < length :
				self.maxLength = length
			self.totalLength += length
			
			self.nbElements[k] = length
			self.classNumbers[k] = i
			self.onehots[k] = numpy.zeros(len(self.classSets))
			self.onehots[k][i] = 1
			i += 1

		subsetSize = self.totalLength
		self.subsets = {
			"input" : numpy.zeros((subsetSize, self.inputSize)),
			"classNumber" : numpy.zeros(subsetSize),
			"onehot" : numpy.zeros( (subsetSize, len(self.classSets)) )
		}

		self._mustReroll = True

	def reroll(self) :
		"""shuffle subsets"""
		offset = 0
		for k, v in self.classSets.iteritems() :
			start, end = offset, offset+self.minLength
			indexes = numpy.random.randint(0, self.nbElements[k], self.minLength)
			self.subsets["input"][start : end] = v[indexes]
			self.subsets["classNumber"][start : end] = self.classNumbers[k]
			self.subsets["onehot"][start : end] = self.onehots[k]
			offset += self.minLength

		size = len(self.subsets["input"])
		indexes = numpy.random.randint(0, size, size)
		for k, v in self.subsets.iteritems() :
			self.subsets[k] = self.subsets[k][indexes]

		self._mustReroll = False

	def get(self, subset, i, size) :
		"""Returns n element from a subset, starting from position i"""
		if self._mustReroll :
			self.reroll()

		return self.subsets[subset][i:i+size]

	def getAll(self, subset) :
		"""return all the elements from a subset"""
		if self._mustReroll :
			self.reroll()

		return self.subsets[subset]

	def getHandle(self, subset) :
		"""returns a DatasetHandle(self, subset)"""
		if subset in self.subsets :
			return DatasetHandle(self, subset)

	def __len__(self) :
		return self.minLength

class DatasetMapper(object):
	"""a DatasetMapper maps Input and Output layer to the data they must receive.
	It's much less complicated than it sounds cf. doc for map(), you can ignore the others.
	The DatasetMapper is intended to be used with a Trainer"""

	def __init__(self):
		self.datasets = set()
		self.maps = {}
		self.inputLayers = []
		self.outputLayers = []
		self.minLength = float('inf')

	def mapInput(self, layer, setHandle) :
		"""
		Maps an input or an output layer to Dataset's subset::
			
			i = InputLayer(...)

			trainSet = RandomSeries(images = train_set[0], classes = train_set[1])
			
			DatasetMapper = dm
			dm.map(i, trainSet.images)
		"""
		
		self.inputLayers.append(layer)
		self.maps[layer] = ( (layer.name, setHandle), )

		self.datasets.add(setHandle.dataset)
		if len(setHandle.dataset) < self.minLength : 
			self.minLength = len(setHandle.dataset)

	def mapOutput(self, layer, setHandle, inputName = 'targets') :
		"""
		Maps an input or an output layer to Dataset's subset::
			
			o = OutputLayer(...)

			trainSet = RandomSeries(images = train_set[0], classes = train_set[1])
			
			DatasetMapper = dm
			dm.map(o, trainSet.classes, "targets")
		
		inputName: train and test functions often need addictional inputs such as the targets. With this
		parameter you can associate one of these additional inputs (as defined in a theano function) to
		a dataset. This argument is optional, the default value is 'targets' which should work for all 
		out of the box Mariana stuff. 
		"""
		
		self.outputLayers.append(layer)
		k = (inputName, setHandle)

		try :
			self.maps[layer].append(k)
		except KeyError :
			self.maps[layer] = [ k ]
		
		self.datasets.add(setHandle.dataset)
		if len(setHandle.dataset) < self.minLength : 
			self.minLength = len(setHandle.dataset)

	def reroll(self) :
		"""rerolls all datasets"""
		for d in self.datasets :
			d.reroll()

	def getBatch(self, i, size, layerList=None) :
		"""
		Returns a dictionary::
			
				layer.name => elements

		:param list layerList: The list of layers for which to return data. If None, will return for all layers.
		"""
		if i > self.minLength :
			raise IndexError("index i '%s', out of range '%s'" % (i, size))

		res = {}
		if layerList is None :
			layers = self.maps.keys()
		else :
			layers = layerList

		for layer in layers :
			for name, handle in self.maps[layer] :
				# handle = self.maps[layer]
				res[name] = handle.dataset.get(handle.subset, i, size)

		return res

	def getAll(self, layerList=None) :
		"""Same as getBatch() but returns the totality of the subset for each layer

		:param list layerList: The list of layers for which to return data. If None, will return for all layers.
		"""

		if layerList is None :
			layers = self.maps.keys()
		else :
			layers = layerList

		res = {}
		for layer in layers :
			for name, handle in self.maps[layer] :
				# handle = self.maps[layer]
				res[name] = handle.dataset.getAll(handle.subset)

		return res

	def __len__(self) :
		return self.minLength

