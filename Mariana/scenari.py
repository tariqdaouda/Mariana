import theano
import theano.tensor as tt

__all__ = ["LearningScenario_ABC", "Fixed", "GradientDescent", "MomentumGradientDescent", "GradientFloor"]

class LearningScenario_ABC(object):
 	"""
 	This is the interface all scenarii must expose
 	"""
	def __init__(self, *args, **kwargs):
		self.name = self.__class__.__name__

	def getUpdates(self, layer, cost) :
		"""return the updates for the parameters of layer. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

	def update(self, trainer) :
		"""reads the store of the trainer and does something to the parameters. By default this function does nothing"""
		pass

class Fixed(LearningScenario_ABC):
	"No learning, the layer weights stay fixed"
 	def __init__(self):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__

 	def getUpdates(self, layer, cost) :
		return []

class GradientDescent(LearningScenario_ABC):
	"The GradientDescent scenarios has a fixed learning rate."
 	def __init__(self, lr):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__
 		self.lr = lr
 		self.hyperParameters = ["lr"]

 	def getUpdates(self, layer, cost) :
 		updates = []
 		if self.lr > 0 :
	 		for param in layer.getParams() :
	 			gparam = tt.grad(cost, param)
		 		updates.append((param, param - self.lr * gparam))

		return updates

class MomentumGradientDescent(LearningScenario_ABC):
	"The MomentumGradientDescent scenarios has a fixed learning rate and a fixed momentum."
 	def __init__(self, lr, momentum):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__
 		self.lr = lr
 		self.momentum = momentum
 		self.hyperParameters = ["lr", "momentum"]

 	def getUpdates(self, layer, cost) :
 		updates = []
 		if self.lr > 0 :
	 		for param in layer.getParams() :
	 			gparam = tt.grad(cost, param)
		 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
				updates.append((param, param - self.lr * momentum_param))

		return updates

class GradientFloor(LearningScenario_ABC):
	"A gradient descent that only propagates the gradient if it's supperior to a floor value."
 	def __init__(self, lr, momentum, floor):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__
 		
 		self.lr = lr
 		self.momentum = momentum
 		self.floor = floor
 		self.hyperParameters = ["lr", "momentum", "floor"]

 	def getUpdates(self, layer, cost) :
 		updates = []
 		if self.lr > 0 :
	 		for param in layer.getParams() :
	 			g = tt.grad(cost, param)
	 			gparam = tt.switch( tt.abs_(g) > self.floor, g, 0.)

		 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
				updates.append((param, param - self.lr * momentum_param))

		return updates