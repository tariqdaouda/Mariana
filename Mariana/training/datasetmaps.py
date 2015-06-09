import Mariana.settings as MSET
import numpy, random

# r = RandomSynchronizedLists(numbers = numbers, classes = classes)
# ds.map(i, r.numbers)
# ds.map(o, r.classes)

# s = SynchronizedClassSets(cars = cars, planes = planes)
# ds.map(i, s.inputs)
# ds.map(o, s.classNumber)

class DatesetHandle(object) :
	def __init__(self, dataset, subset) :
		self.dataset = dataset
		self.subset = subset

	def __repr__(self) :
		return "<Handle %s.%s>" % (self.dataset.__class__.__name__, self.subset)

class Dataset_ABC(object) :
	
	def getOne(self) :
		pass

	def getAll(self) :
		pass

	def __len__(self) :
		pass

	def _getHandle(self, subset) :
		raise NotImplemented("Should be implemented in child")

	def __getattr__(self, k) :
		g = object.__getattribute__(self, "_getHandle")
		res = g(k)
		if res :
			return res
		raise AttributeError("No attribute or dataset by the name of '%s'" % k)

class SynchronizedLists(Dataset_ABC) :
	"""Returns the examples sequentially. All lists must have the same length"""
	def __init__(self, **kwargs) :
		self.lists = {}
		self.length = 0
		self.pos = 0
		for k, v in kwargs.iteritems() :
			if self.length == 0 :
				self.length = len(v)
			else :
				if len(v) != self.length :
					raise ValueError("All lists must have the same length, previous list had a length of '%s', list '%s' has a length of '%s" % (self.length, k, len(v)))
			self.lists[k] = v

	def getOne(self) :
		if self.pos >= self.length :
			raise StopIteration("End of set")
		
		res = {}
		for k, v in self.lists.iteritems() :
			res[k] = v[self.pos]
		
		self.pos += 1
		return res

	def getAll(self) :
		return self.lists

	def _getHandle(self, subset) :
		if subset in self.lists :
			return DatesetHandle(self, subset)
		return None

	def __len__(self) :
		return self.length

class RandomSynchronizedLists(SynchronizedLists) :
	"""picks the examples at random"""
	def __init__(self, **kwargs) :
		SynchronizedLists.__init__(self, **kwargs)

	def getOne(self) :
		r = random.randint(0, self.length-1)
		res = {}
		for k, v in self.lists.iteritems() :
			res[k] = v[r]

		return res

class SynchronizedClassSets(Dataset_ABC) :
	""""""
	def __init__(self, **kwargs) :
		self.sets = {}
		self.handles = set(["inputs", "className", "classNumber", "onehot"])

		self.minLength = float('inf')
		self.maxLength = 0
		self.totalLength = 0
		for k, v in kwargs.iteritems() :
			if len(v) < self.minLength :
				self.minLength = len(v)
			if self.maxLength < len(v) :
				self.maxLength = len(v)
			self.sets[k] = v
			self.totalLength += len(v)

		self.minLength = int(self.minLength)

		self.likelihoods = {}
		self.classNumbers = {}
		self.onehots = {}
		i = 0
		for k, v in self.sets.iteritems() :
			self.likelihoods[k] = len(self.sets[k])/float(self.totalLength)
			self.classNumbers[k] = i
			self.onehots[k] = numpy.zeros(len(self.sets))
			self.onehots[k][i] = 1
			i += 1

	def setLikelihoods(self, **kwargs) :
		if len(self.kwargs) != len(self.sets) :
			raise ValueError("the number of arguments should be the same as the number of registred sets")

		s = 0
		for k, v in self.kwargs.iteritems() :
			if k not in self.sets :
				raise ValueError("there's no registered set by the name of %s'" % k)
			self.likelihood[k] = v

		if s != 1 :
			raise ValueError("the sum of all likelihoods must be 1, got '%s'" % s)

	def getOne(self):
		r = random.random()
		offset = 0
		for k, v in self.sets.iteritems() :
			if self.likelihoods[k] < (r + offset) :
				return {"inputs": random.choice(v), "className": k, "classNumber": self.classNumbers[k], "onehot": self.onehots[k]}
			offset += self.likelihoods[k]

	def getAll(self) :
		res = {"inputs": [], "className": [], "classNumber": [], "onehot": []}
		for i in xrange(self.minLength) :
			r = random.random()
			offset = 0
			for k, v in self.sets.iteritems() :
				if self.likelihoods[k] < (r + offset) :
					res["inputs"].append(random.choice(v))
					res["className"].append(k)
					res["classNumber"].append(self.classNumbers[k])
					res["onehot"].append(self.onehots[k])
					break
				offset += self.likelihoods[k]
		return res

	def _getHandle(self, subset) :
		if subset in self.handles :
			return DatesetHandle(self, subset)

	def __len__(self) :
		return self.minLength

	def __len__(self) :
		return self.minLength

class DatasetMapper(object):
	"""docstring for DatasetMapper"""
	def __init__(self):
		self.datasets = set()
		self.maps = {}
		self.inputLayers = []
		self.outputLayers = []
		self.minLength = float('inf')
		
		self.currentData = {}

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
		self.currentData = {}
		tmpSets = {}
		for i in xrange(self.minLength) :
			for dataset in self.datasets :
				try :
					tmpSets[dataset].append(dataset.getOne())
				except KeyError :
					tmpSets[dataset] = [dataset.getOne()]

		for i in xrange(self.minLength) :
			for layer, handle in self.maps.iteritems() :
				try :
					self.currentData[layer.name].append(tmpSets[handle.dataset][i][handle.subset])
				except KeyError :
					self.currentData[layer.name] = [tmpSets[handle.dataset][i][handle.subset]]

	def getBatch(self, i, size) :
		if len(self.currentData) == 0 :
			self.reroll()

		if i > self.minLength :
			raise IndexError("index i '%s', out of range '%s'" % (i, size))

		res = {}
		for k, v in self.currentData.iteritems() :
			res[k] = v[i : i + size]
		return res

	def getAll(self) :
		if len(self.currentData) == 0 :
			self.reroll()

		return self.currentData

	def __len__(self) :
		return self.minLength

