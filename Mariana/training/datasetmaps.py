import Mariana.settings as MSET
import numpy, random

class DatesetHandle(object) :
	def __init__(self, dataset, subset) :
		self.dataset = dataset
		self.subset = subset

	def __repr__(self) :
		return "<Handle %s.%s>" % (self.dataset.__class__.__name__, self.subset)

class Dataset_ABC(object) :
	
	def get(self, subset, n) :
		raise NotImplemented("Should be implemented in child")

	def getAll(self, subset) :
		raise NotImplemented("Should be implemented in child")

	def reroll(self) :
		raise NotImplemented("Should be implemented in child")

	def _getHandle(self, subset) :
		raise NotImplemented("Should be implemented in child")

	def __len__(self) :
		raise NotImplemented("Should be implemented in child")

	def __getattr__(self, k) :
		g = object.__getattribute__(self, "_getHandle")
		res = g(k)
		if res :
			return res
		raise AttributeError("No attribute or dataset by the name of '%s'" % k)

class Series(Dataset_ABC) :
	"""Returns the examples sequentially. All lists must have the same length"""
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
		pass

	def get(self, subset, i, n) :
		return self.lists[subset][i:i + n]

	def getAll(self, subset) :
		return self.lists[subset]

	def _getHandle(self, subset) :
		if subset in self.lists :
			return DatesetHandle(self, subset)
		return None

	def __len__(self) :
		return self.length

	def __repr__(self) :
		return "<%s len: %s, nbLists: %s>" % (self.__class__.__name__, self.length, len(self.lists))

class RandomSeries(Series) :
	"""This is very much like Series, but examples are randomly sampled"""
	def __init__(self, **kwargs) :
		Series.__init__(self, **kwargs)
		self.reroll()

	def reroll(self) :
		indexes = numpy.random.randint(0, self.length, self.length)
		for k in self.lists :
			self.lists[k] = self.lists[k][indexes]

class ClassSets(Dataset_ABC) :
	"""Pass it one set for each class and it will take care of randomly sampling them with replacement.
	All class sets must have at least two elements"""
	def __init__(self, **kwargs) :
		self.classSets = {}
		
		self.minLength = float('inf')
		self.maxLength = 0
		self.totalLength = 0
		self.inputSize = 0
		for k, v in kwargs.iteritems() :
			if len(v) < self.minLength :
				self.minLength = len(v)
			if self.maxLength < len(v) :
				self.maxLength = len(v)
			self.classSets[k] = numpy.asarray(v)
			if len(self.classSets[k].shape) < 2 :
			 	self.classSets[k] = self.classSets[k].reshape(self.classSets[k].shape[0], 1)

			self.totalLength += len(v)
			
			try :
				inputSize = len(v[0])
			except :
				inputSize = 1

			if self.inputSize == 0 :
				self.inputSize = inputSize
			elif self.inputSize != inputSize :
				raise ValueError("All class elements must have the same size. Got '%s', while the previous value was '%s'" % ( len(v[0]), self.inputSize ))
				
		self.minLength = int(self.minLength)
		if self.minLength < 2 :
			raise ValueError('All class sets must have at least two elements')

		subsetSize = self.totalLength-len(self.classSets) #-1 for each class set
		self.subsets = {
			"input" : numpy.zeros( (subsetSize, self.inputSize) ),
			"classNumber" : numpy.zeros( (subsetSize, 1) ),
			"onehot" : numpy.zeros( (subsetSize, len(self.classSets)) )
		}

		self.classNumbers = {}
		self.onehots = {}
		self.nbElements = {}
		i = 0
		for k, v in self.classSets.iteritems() :
			self.nbElements[k] = len(self.classSets[k]) -1 #-1 to simulate a sampling with replacement
			self.classNumbers[k] = i
			self.onehots[k] = numpy.zeros(len(self.classSets))
			self.onehots[k][i] = 1
			i += 1

		self._mustReroll = True

	def reroll(self) :
		offset = 0
		for k, v in self.classSets.iteritems() :
			start, end = offset, offset+self.nbElements[k]
			indexes = numpy.random.randint(0, self.nbElements[k], self.nbElements[k])
			self.subsets["input"][start : end] = v[indexes]
			self.subsets["classNumber"][start : end] = self.classNumbers[k]
			self.subsets["onehot"][start : end] = self.onehots[k]
			offset += self.nbElements[k]

		size = len(self.subsets["input"])
		indexes = numpy.random.randint(0, size, size)
		for k, v in self.subsets.iteritems() :
			self.subsets[k] = self.subsets[k][indexes]

		self._mustReroll = False

	def setEvenLikelihoods(self) :
		"""sample so that all classes have the same chances of appearing"""
		for k, v in self.classSets.iteritems() :
			self.nbElements[k] = self.minLength -1 #-1 to simulate a sampling with replacement

		subsetSize = self.minLength-len(self.classSets) #-1 for each class set
		self.subsets = {
			"input" : numpy.zeros( (subsetSize, self.inputSize) ),
			"classNumber" : numpy.zeros( (subsetSize, 1) ),
			"onehot" : numpy.zeros( (subsetSize, len(self.classSets)) )
		}

		self._mustReroll = True

	def get(self, subset, i, size) :
		if self._mustReroll :
			self.reroll()

		return self.subsets[subset][i:i+size]

	def getAll(self, subset) :
		if self._mustReroll :
			self.reroll()

		return self.subsets[subset]

	def _getHandle(self, subset) :
		if subset in self.subsets :
			return DatesetHandle(self, subset)

	def __len__(self) :
		return self.totalLength

class DatasetMapper(object):
	"""docstring for DatasetMapper"""
	def __init__(self):
		self.datasets = set()
		self.maps = {}
		self.inputLayers = []
		self.outputLayers = []
		self.minLength = float('inf')
		
		self.res = {}

	def map(self, layer, setHandle) :
		if layer.type == MSET.TYPE_OUTPUT_LAYER :
			self.outputLayers.append(layer)
		elif layer.type == MSET.TYPE_INPUT_LAYER :
			self.inputLayers.append(layer)
		else :
			raise ValueError("Only input and output layers can be mapped")

		self.maps[layer] = setHandle
		self.datasets.add(setHandle.dataset)
		if len(setHandle.dataset) < self.minLength : 
			self.minLength = len(setHandle.dataset)

	def reroll(self) :
		for d in self.datasets :
			d.reroll()

	def getBatch(self, i, size) :
		if i > self.minLength :
			raise IndexError("index i '%s', out of range '%s'" % (i, size))

		for layer, handle in self.maps.iteritems() :
			self.res[layer.name] = handle.dataset.get(handle.subset, i, size)

		return self.res

	def getAll(self) :
		for layer, handle in self.maps.iteritems() :
			self.res[layer.name] = handle.dataset.getAll(handle.subset)
		return self.res

	def __len__(self) :
		return self.minLength

