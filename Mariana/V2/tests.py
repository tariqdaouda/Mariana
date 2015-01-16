import unittest

from layers import *
import Mariana.costs as MC
import theano.tensor as tt
import numpy as N

class MLPTests(unittest.TestCase):

	def setUp(self) :
		pass

	def tearDown(self) :
		pass

	def test_xor(self) :
		i = Input(2)
		h = Hidden(4, activation = tt.tanh)
		o = Output(2, lr = 0.1, activation = tt.nnet.softmax, costFct = MC.negativeLogLikelihood, name = "out")
		mlp = i > h > o

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
		
		self.assertEqual(mlp.predict( [ ins[0] ] )["out"][0], 0 )
		self.assertEqual(mlp.predict( [ ins[1] ] )["out"][0], 1 )
		self.assertEqual(mlp.predict( [ ins[2] ] )["out"][0], 1 )
		self.assertEqual(mlp.predict( [ ins[3] ] )["out"][0], 0 )

if __name__ == '__main__' :
	unittest.main()