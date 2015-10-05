import theano.tensor as tt

def sigmoid(x):
	"""
	.. math::

		1/ (1/ + exp(-x))"""
	return tt.nnet.sigmoid(x)
	
def tanh(x):
	"""
	.. math::

		tanh(x)"""
	return tt.tanh(x)
	
def reLU(x):
	"""
	.. math::

		max(0, x)"""
	#dp not replace by theano's relu. It works bad with nets that have multiple outputs
	return tt.maximum(0., x)
	
def softmax(x):
	"""Softmax to get a probabilistic output"""
	return tt.nnet.softmax(x)