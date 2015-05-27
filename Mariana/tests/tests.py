import unittest

import Mariana.layers as ML
import Mariana.decorators as dec
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.activations as MA

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
		ls = MS.DefaultScenario(lr = 0.1, momentum = 0)
		cost = MC.NegativeLogLikelihood()

		i = ML.Input(2, 'inp')
		h = ML.Hidden(4, activation = MA.tanh, decorators = [dec.GlorotTanhInit()], regularizations = [MR.L1(0), MR.L2(0)])
		o = ML.SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")

		mlp = i > h > o

		self.xor_ins = N.array(self.xor_ins)
		self.xor_outs = N.array(self.xor_outs)
		for i in xrange(1000) :
			ii = i%len(self.xor_ins)
			mlp.train(o, inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ] )
		
		return mlp

	# @unittest.skip("skipping")
	def test_xor(self) :
		mlp = self.trainMLP_xor()
		o = mlp.outputs.values()[0]
		
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[3] ] )[0], 0 )

	# @unittest.skip("skipping")
	def test_save_load(self) :
		import cPickle, os

		mlp = self.trainMLP_xor()
		mlp.save("test_save")
		mlp2 = cPickle.load(open('test_save.mariana.pkl'))


		o = mlp.outputs.values()[0]
		self.assertEqual(mlp2.classify( o, inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp2.classify( o, inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp2.classify( o, inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp2.classify( o, inp = [ self.xor_ins[3] ] )[0], 0 )
		
		os.remove('test_save.mariana.pkl')

	# @unittest.skip("skipping")
	def test_composite(self) :
		ls = MS.DefaultScenario(lr = 0.1, momentum = 0)
		cost = MC.NegativeLogLikelihood()

		inp = ML.Input(2, 'inp')
		h1 = ML.Hidden(2, activation = MA.tanh, name = "h1")
		h2 = ML.Hidden(2, activation = MA.tanh, name = "h2")
		o = ML.SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")
		c = ML.Composite(name = "Comp")
		
		inp > h1 > c
		inp > h2 > c
		mlp = c > o
	
		self.xor_ins = N.array(self.xor_ins)
		self.xor_outs = N.array(self.xor_outs)
		for i in xrange(1000) :
			ii = i%len(self.xor_ins)
			mlp.train(o, inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ])
		
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[0] ] )[0], 0 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[1] ] )[0], 1 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[2] ] )[0], 1 )
		self.assertEqual(mlp.classify( o, inp = [ self.xor_ins[3] ] )[0], 0 )
		
if __name__ == '__main__' :
	unittest.main()
