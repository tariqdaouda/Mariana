# import Mariana.costs as MC
import theano
import theano.tensor as tt

class LearningScenario_ABC(object):
 	"""This class allow to specify specific learning scenarios for layers independetly"""
	def __init__(self, *args, **kwargs):
		self.name = self.__class__.__name__

	def getUpdates(self, layer, cost) :
		"""return the updates for the parameters of layer. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

class DefaultScenario(LearningScenario_ABC):
	"The default scenarios has a fixed learning rate and a fixed momentum"
 	def __init__(self, lr, momentum):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__
 		self.lr = lr
 		self.momentum = momentum
 		self.hyperParameters = ["lr", "momentum"]

 	def getUpdates(self, layer, cost) :
 		updates = []
 		if self.lr > 0 :
	 		for param in layer.params :
	 			gparam = tt.grad(cost, param)
		 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
				updates.append((param, param - self.lr * momentum_param))

		return updates

class Fixed(LearningScenario_ABC):
	"No learning, the layer weights stay fixed"
 	def __init__(self):
 		super(LearningScenario_ABC, self).__init__()
 		self.name = self.__class__.__name__

 	def getUpdates(self, layer, cost) :
		return []

class GradientFloor(LearningScenario_ABC):
	"On propagates the garidient of its absolute value is above floor"
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
	 		for param in layer.params :
	 			g = tt.grad(cost, param)
	 			gparam = tt.switch( tt.abs_(g) > self.floor, g, 0.)

		 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
				updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
				updates.append((param, param - self.lr * momentum_param))

		return updates