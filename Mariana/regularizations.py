class Regularization(object) :

	def __init__(self, factor, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.factor = factor
		self.layers = []
		# self.hyperParameters = []

	def _registerLayer(self, layer) :
		self.layers.append(layer)

	def _formula(self, layer) :
		"""The cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

class L1(Regularization) :
	
	def _formula(self, layer) :
		s = 0
		for l in self.layers :
			if l.W is not None :
				s += abs(l.W).sum()
		return self.factor * ( abs(outputLayer.W).sum() + s )

class L2(Regularization) :
	
	def _formula(self, layer) :
		s = 0
		for l in self.layers :
			if l.W is not None :
				s += (l.W**2).sum()
		return self.factor * ( abs(outputLayer.W).sum() + s )