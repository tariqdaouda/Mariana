import unittest

from Mariana.layers import *
from Mariana.rules import *
import Mariana.costs as MC
import theano.tensor as tt
import numpy as N

class MLPTests(unittest.TestCase):

	def setUp(self) :
		self.xor_ins = [
			[0, 0],
			[0, 1],
			[1, 0],
			[1, 1]
		]

		self.xor_outs = [0, 1, 1, 0]

	def tearDown(self) :
		pass

	def trainMLP_xor(self) :
		ls = DefaultScenario(lr = 0.1, momentum = 0)
		cost = NegativeLogLikelihood(l1 = 0, l2 = 0)

		i = Input(2, 'inp')
		h = Hidden(4, activation = tt.tanh)
		o = SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")

		mlp = i > h > o

		self.xor_ins = N.array(self.xor_ins)
		self.xor_outs = N.array(self.xor_outs)
		for i in xrange(1000) :
			ii = i%len(self.xor_ins)
			mlp.train("out", inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ] )
		
		return mlp

	# @unittest.skip("skipping")
	def test_xor(self) :
		mlp = self.trainMLP_xor()
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[3] ] )[0], 0 )

	# @unittest.skip("skipping")
	def test_save_load(self) :
		import cPickle, os

		mlp = self.trainMLP_xor()
		mlp.save("test_save")
		mlp2 = cPickle.load(open('test_save.mariana.pkl'))

		self.assertEqual(mlp2.classify( "out", inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp2.classify( "out", inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp2.classify( "out", inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp2.classify( "out", inp = [ self.xor_ins[3] ] )[0], 0 )
		
		os.remove('test_save.mariana.pkl')

	# @unittest.skip("skipping")
	def test_composite(self) :
		ls = DefaultScenario(lr = 0.1, momentum = 0)
		cost = NegativeLogLikelihood(l1 = 0, l2 = 0)

		inp = Input(2, 'inp')
		h1 = Hidden(2, activation = tt.tanh, name = "h1")
		h2 = Hidden(2, activation = tt.tanh, name = "h2")
		o = SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")
		c = Composite(name = "Comp")
		
		inp > h1 > c
		inp > h2 > c
		mlp = c > o
	
		self.xor_ins = N.array(self.xor_ins)
		self.xor_outs = N.array(self.xor_outs)
		for i in xrange(1000) :
			ii = i%len(self.xor_ins)
			mlp.train("out", inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ])
		
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp.classify( "out", inp = [ self.xor_ins[3] ] )[0], 0 )


if __name__ == '__main__' :
	unittest.main()
