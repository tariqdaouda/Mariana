import Mariana.layers as layers
# import Mariana.costs as MC
import theano
import theano.tensor as tt

def negativeLogLikelihood(targets, outputs) :
	"""cost fct for softmax"""
	cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
	return cost

def crossEntropy(targets, outputs) :
	cost = -tt.nnet.binary_crossentropy(targets, outputs).mean()
	return cost

def meanSquaredError(targets, outputs) :
	"""The all time classic"""
	cost = -tt.mean( tt.dot(outputs, targets) **2 )
	return cost

class LearningScenario_ABC(object):
 	"""This class allow to specify specific learning scenarios for layers independetly"""
	def __init__(self, *args, **kwargs):
		pass

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
 		# super(LearningRules, self).__init__()
 		self.lr = lr
 		self.momentum = momentum

 	def getUpdates(self, layer, cost) :
 		updates = []
 		for param in layer.params :
 			# print "\top", layer, param, self.lr
 			gparam = tt.grad(cost, param)
	 		momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
			updates.append((momentum_param, self.momentum * momentum_param + (1-self.momentum)*gparam))
			updates.append((param, param - self.lr * momentum_param))	 		

		return updates

	def update(self, *args, **kwargs) :
		pass

class Cost_ABC(object) :
	"""This allows to create custom costs by adding stuff such as regularizations"""

	def __init__(self, *args, **kwargs) :
		pass

	@classmethod
	def getL1(cls, l1, outputLayer) :
		"""returns the l1 regularization cost taking into acocunt all the depencies of outputLayer"""
		s1 = 0
		for l in outputLayer.dependencies.itervalues() :
			if l.W is not None :
				s1 += abs(l.W).sum()
		return l1 * ( abs(outputLayer.W).sum() + s1 )

	@classmethod
	def getL2(cls, l2, outputLayer) :
		"""returns the l2 regularization cost taking into acocunt all the depencies of outputLayer"""
		s2 = 0
		for l in outputLayer.dependencies.itervalues() :
			if l.W is not None :
				s2 += (l.W**2).sum()
		return l2 * ( (outputLayer.W**2).sum() + s2 )

	def getCost(self, layer, cost) :
		"""returns the cost function. Must be implemented in child"""
		raise NotImplemented("Must be implemented in child")

	# def update(self) :
	# 	"""this function is called automatically called before each train() call.
	# 	It works the same as for scenarii, the default version does nothing, and yes
	# 	you can modify your regularization parameters as the learning goes"""
	# 	pass

class NegativeLogLikelihood(Cost_ABC) :
	def __init__(self, l1 = 0, l2 = 0) :
		self.l1 = l1
		self.l2 = l2
		self.costFct = negativeLogLikelihood

	def getCost(self, outputLayer) :
		L1 = Cost_ABC.getL1(self.l1, outputLayer)		
		L2 = Cost_ABC.getL2(self.l2, outputLayer)		
		return self.costFct(outputLayer.target, outputLayer.outputs) + L1 + L2


class CrossEntropy(Cost_ABC) :
	def __init__(self, l1 = 0, l2 = 0) :
		self.l1 = l1
		self.l2 = l2
		self.costFct = crossEntropy

	def getCost(self, outputLayer) :
		L1 = Cost_ABC.getL1(self.l1, outputLayer)		
		L2 = Cost_ABC.getL2(self.l2, outputLayer)		
		return self.costFct(outputLayer.target, outputLayer.outputs) + L1 + L2


