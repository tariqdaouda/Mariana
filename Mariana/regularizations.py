class SingleLayerRegularizer(object) :

	def __init__(self, factor, *args, **kwargs) :
		self.name = self.__class__.__name__
		self.factor = factor
		self.hyperparameters = ["factor"]

	def getFormula(self, layer) :
		"""This function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

class L1(SingleLayerRegularizer) :
	
	def getFormula(self, layer) :
		return self.factor * ( abs(layer.W).sum() )

class L2(SingleLayerRegularizer) :
	
	def getFormula(self, layer) :
		return self.factor * ( abs(layer.W).sum() )