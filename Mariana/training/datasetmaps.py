import Mariana.settings as MSET
import numpy, random, time

class ListSet(object) :
	def __init__(self, values, name = None) :
		self.values = values
		if name is not None :
			self.name = name
		else :
			self.name = time.clock() + random.randint(0, 100)

	def getAll(self) :
		return self.values
	
	def __getitem__(self, i) :
		return self.values[i]

	def __len__(self) :
 		return len(self.values)

 	def __repr__(self) :
 		return "<Mariana list set len: %d>" % len(self)

class _InnerSet(object) :
	"""An inner set of a ClassSuperset"""

	def __init__(self, values, classNumber) :
		self.values = values
		self.classNumber = classNumber
		self.targets = numpy.ones(len(self.values), dtype = "int32") * self.classNumber
		self.offset = 0

	def randomPos(self, bufferLen) :
		if bufferLen == len(self) :
			self.offset = 0
		elif bufferLen > len(self) :
			raise ValueError("bufferLen can not be > len(self): %s > %s" % (bufferLen, len(self)))
		else :
			self.offset = random.randrange( 0, (len(self) - bufferLen) )

	def getAll(self) :
		return ( self.values[self.offset:], self.targets[self.offset:] )

	def __len__(self) :
		return len(self.values)

	def __getitem__(self, i) :
		if i.__class__ == slice :
			s = slice( self.offset + i.start, self.offset + i.stop, i.step )
			# print s, self.targets[s], i, len(self.values),  self.offset
			return ( self.values[s], self.targets[s] )
		return ( self.values[self.offset + i], self.targets[self.offset + i] )

class ClassSuperset(object) :

	def __init__(self, name = None) :
		self.sets = []
		self.likelihoods = []
		
		self.likelihoodsSum = 0

		self.maxLen = 0
		self.minLen = 0
		self.totalLen = 0
		self.smallestSetLen = 0

		self.nbClasses = 0
		self.runExamples = [] 
		self.runTargets = []
		self.runIds = []

		self.currentInps = []
		self.currentOuts = []

		if name is None :
			self.name = time.clock() + random.randint(0, 100)
		else :
			self.name = name

	def setMinLen(self, v) :
		if v > self.smallestSetLen :
			raise ValueError("Min length cannot be smaller that le length of the smallest set: %d" % self.smallestSetLen)
		self.minLen = v

	def add(self, lst, likelihood = -1) :
		aSet = _InnerSet(lst, self.nbClasses)
		self.sets.append( aSet )
		
		self.totalLen += len(aSet)

		if len(aSet) > self.maxLen :
			self.maxLen = len(aSet)

		if self.minLen == 0 or len(aSet) < self.minLen :
			self.minLen = len(aSet)
			self.smallestSetLen = self.minLen

		self.nbClasses += 1

		if likelihood > 0 :
			self.likelihoodsSum += likelihood
			if self.likelihoodsSum > 1 :
				raise ValueError("The sum of the likelihoods cannot be > 1")

			self.likelihoods.append(self.likelihoodsSum)

	def shuffle(self) :
		for s in self.sets :
			s.randomPos(self.minLen)

	def getAll(self) :
		examples = []
		targets = []

		for s in self.sets :
			g = s.getAll()
			examples.extend(g[0])
			targets.extend(g[1])

		return (examples, targets)

	def __getitem__(self, i) :
		if self.likelihoodsSum == 0 :
			for s in self.sets :
				self.likelihoodsSum += 1./len(self.sets)#float(len(s)) / self.totalLen
				self.likelihoods.append( self.likelihoodsSum )

		elif self.likelihoodsSum > 0 and self.likelihoodsSum < 1 :
				raise ValueError("All likelihoods must sum to 1, the current value is %s" % self.likelihoodsSum)

		r = random.random()
		for j in xrange(len(self.likelihoods)) :
			if r <= self.likelihoods[j] :
				return self.sets[j][i]
		
		raise ValueError("Invalid likelihoods : %s (sum: %s)" % (self.likelihoods, sum(self.likelihoods)) )

	def __len__(self) :
 		return self.minLen

class DatasetMapper(object):
 	"""docstring for DatasetMapper"""

 	def __init__(self):
 		self.inputSets = {}
 		self.outputSets = {}
 		self.layerNames = set()
 		self.outputLayerNames = set()
 		self.sets = {}

		self.syncedLayers = {}
		
		self.minLen = 0
		self.runIds = []
		self.mustInit = True

	def mapInput(self, lst, layer) :
		if layer.name in self.layerNames :
			raise ValueError("There is already a registered layer by the name of: '%s'" % (layer.name)) 

		if lst.__class__ not in [ClassSuperset, ListSet] :
			aSet = ListSet(lst)
		else :
			aSet = lst

		if layer.type == MSET.TYPE_INPUT_LAYER :
			self.inputSets[aSet.name] = layer
		else :
			raise ValueError("Only input layers are allowed")
 
		self.layerNames.add(layer.name)
		self.sets[aSet.name] = aSet
		
		if self.minLen == 0 or len(aSet) < self.minLen :
			self.minLen = len(aSet)

	def mapOutput(self, lst, layer) :
		if layer.name in self.layerNames :
			raise ValueError("There is already a registered layer by the name of: '%s'" % (layer.name))

		if lst.__class__ is not ClassSuperset :
			aSet = ListSet(lst)
		else :
			aSet = lst

		if layer.type == MSET.TYPE_OUTPUT_LAYER :
			self.outputSets[aSet.name] = layer
		else :
			raise ValueError("Only output layers are allowed")
		
		self.layerNames.add(layer.name)
		self.outputLayerNames.add(layer.name)
		self.sets[aSet.name] = aSet
		
		if self.minLen == 0 or len(aSet) < self.minLen :
			self.minLen = len(aSet)

	def syncLayers(self, refLayer, layer) :
		"""Ensures that all layers in 'layers' receive the same data as refLayer"""

		if refLayer.name not in self.layerNames :
			raise ValueError("There's no registered refLayer by the name '%s'" % refLayer.name)
		
		if layer.name in self.layerNames :
			raise ValueError("There is already a registered layer by the name of: '%s'" % (layer.name))
		
		try :
			self.syncedLayers[refLayer.name].append(layer)
		except KeyError, IndexError:
			self.syncedLayers[refLayer.name] = [layer]
		
		if layer.type == MSET.TYPE_OUTPUT_LAYER :
			self.outputLayerNames.add(layer.name)
		self.layerNames.add(layer.name)

	def shuffle(self) :
		if self.mustInit :
			self._init()

		for s in self.sets.itervalues() :
			if s.__class__ is ClassSuperset : 
				s.setMinLen(self.minLen)
				s.shuffle()
		
		if len(self.runIds) == 0 :
			self.runIds = range(self.minLen)
		random.shuffle(self.runIds)

	def _init(self) :
		self.runIds = range(self.minLen)
		self.mustInit = False

 	def getBatches(self, i, size) :
		"""Returns a random set of examples for each class, all classes have an equal chance of apperance
		regardless of their number of elements. If you want  the limit to be length of the whole set
 		instead of a mini batch you can set size to "all".
 		"""
		if self.mustInit :
			self._init()
			
 		inps = {}
		outs = {}
		ii = self.runIds[i]
		for k, v in self.sets.iteritems() :
			elmt = v[ii: ii+size]
			if v.__class__ is ClassSuperset :
				l = self.inputSets[k]
				inps[l.name] = elmt[0]
				l = self.outputSets[k]
				outs[l.name] = elmt[1]
			else :
				try :
					l = self.inputSets[k]
					inps[l.name] = elmt
				except :
					l = self.outputSets[k]
					outs[l.name] = elmt

		for k, layers in self.syncedLayers.iteritems() :
			for l in layers :
				if l.type == MSET.TYPE_OUTPUT_LAYER :
					try :
						outs[l.name] = inps[k]
					except KeyError :
						outs[l.name] = outs[k]

				elif l.type == MSET.TYPE_INPUT_LAYER :
					try :
						inps[l.name] = inps[k]
					except KeyError :
						outs[l.name] = outs[k]
				else :
					raise ValueError("Synced layer ''%s is neither an input nor an output layer" % l.name)
		
		return (inps, outs)

	def getAll(self) :
		"""Returns the whole batch"""
		if self.mustInit :
			self._init()

		inps = {}
		outs = {}
		
		for k, v in self.sets.iteritems() :
			elmt = v.getAll()
			if v.__class__ is ClassSuperset :
				l = self.inputSets[k]
				inps[l.name] = elmt[0]
				l = self.outputSets[k]
				outs[l.name] = elmt[1]
			else :
				try :
					l = self.inputSets[k]
					inps[l.name] = elmt
				except :
					l = self.outputSets[k]
					outs[l.name] = elmt
		
		for k, layers in self.syncedLayers.iteritems() :
			for l in layers :
				if l.type == MSET.TYPE_OUTPUT_LAYER :
					try :
						outs[l.name] = inps[k]
					except KeyError :
						outs[l.name] = outs[k]
				elif l.type == MSET.TYPE_INPUT_LAYER :
					try :
						inps[l.name] = inps[k]
					except KeyError :
						outs[l.name] = outs[k]
				else :
					raise ValueError("Synced layer ''%s is neither an input nor an output layer" % l.name)
		
		return (inps, outs)

	def getOutputNames(self) :
 		return self.outputLayerNames
 	
 	def __repr__(self) :
 		return "<DatasetMapper len: %s, sets: %s, inputs: %s, outputs: %s>" % (self.minLen, len(self.sets), len(self.inputSets), len(self.outputSets))

	def __len__(self) :
		return self.minLen
