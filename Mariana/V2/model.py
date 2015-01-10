import layers as L

class Model(object) :
	
	def __init__(self,name) :
		self.name = name
		self.layers = {}
		self.inputLayers = {}

	def _registerLayer(self, layer) :
		if layer.name in layers :
			raise KeyError("The model '%s' has already a layer called '%s'" % (self.name, layer.name)
		self.layers[layer.name] = layer
		if layer.__class__ is L.InputLayer :
			self.inputLayers[layer.name] = layer
