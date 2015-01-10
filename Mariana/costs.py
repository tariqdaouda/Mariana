# import numpy
import theano.tensor as tt

def negativeLogLikelihood(y, outputs) :
	"""cost fct for softmax"""
	cost = -tt.mean(tt.log(outputs)[tt.arange(y.shape[0]), y])
	return cost

def meanSquaredError(y, outputs) :
	"""The all time classic"""
	cost = -tt.mean( tt.dot(outputs, y) **2 )
	return cost