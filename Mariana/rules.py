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

	# def update(self) :
	# 	"""this function is called automatically called before each train() call.
	# 	By default it does nothing, but you can it to define crazy learning rules
	# 	such as decreasing the learning rate while exponentially increasing the momentum
	# 	because it's monday"""
	# 	pass

class DefaultScenario(LearningScenario_ABC):
	"The default scenarios has a fixed learning rate and a fixed momentum"
 	def __init__(self, lr, momentum, *args, **kwargs):
 		super(LearningScenario_ABC, self).__init__()
 		self.lr = lr
 		self.momentum = momentum
 		self.hyperParameters = ["lr", "momentum"]

 	def getUpdates(self, layer, cost) :
 		updates = []
 		for param in layer.params :
 			# print "\top", layer, param, self.lr
 			gparam = tt.grad(cost, param)
	 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
			updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
			updates.append((param, param - self.lr * momentum_param))	 		

		return updates

