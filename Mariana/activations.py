import theano.tensor as tt

def sigmoid(x):
	return tt.nnet.sigmoid(x)
	
def tanh(x):
	return tt.tanh(x)
	
def reLU(x):
	return tt.maximum(0., x)

def softmax(x):
	return tt.nnet.softmax(x)