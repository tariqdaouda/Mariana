import unittest

import Mariana.NeuralNet as NN
import Mariana.costs as MC
import theano.tensor as tt
import numpy as N

class MLPTests(unittest.TestCase):

	def setUp(self) :
		pass

	def tearDown(self) :
		pass

	def test_xor(self) :
		mlp = NN.NeuralNet('mlp', lr = 0.1, nbInputs = 2, costFct = MC.negativeLogLikelihood, momentum = 0, l1 = 0, l2 = 0)
		mlp.stackLayer(name = 'hidden1', nbOutputs = 4, activation = tt.tanh)
		mlp.stackLayer(name = 'prediction', nbOutputs = 2, activation = tt.nnet.softmax)

		ins = [
			[0, 0],
			[0, 1],
			[1, 0],
			[1, 1]
		]

		outs = [0, 1, 1, 0]
		ins = N.array(ins)
		outs = N.array(outs)
		for i in xrange(1000) :
			ii = i%len(ins)
			mlp.train([ ins[ ii ] ], [ outs[ ii ] ])
		
		self.assertEqual(mlp.predict( [ ins[0] ] ), 0 )
		self.assertEqual(mlp.predict( [ ins[1] ] ), 1 )
		self.assertEqual(mlp.predict( [ ins[2] ] ), 1 )
		self.assertEqual(mlp.predict( [ ins[3] ] ), 0 )

if __name__ == '__main__' :
	unittest.main()