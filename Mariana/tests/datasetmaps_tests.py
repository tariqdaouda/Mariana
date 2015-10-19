import unittest

import Mariana.layers as ML
import Mariana.layers as ML
import Mariana.decorators as dec
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.activations as MA
import Mariana.training.datasetmaps as MD

import theano.tensor as tt
import numpy

class DastasetMapsTests(unittest.TestCase):

	def setUp(self) :
		pass

	def tearDown(self) :
		pass

	def test_classSets(self) :
		def sample(cls) :
			o = cls.getAll("onehot")
			n = cls.getAll("classNumber")
			p = cls.getAll("input")	

			return o, n, p

		l1 = numpy.arange(10)
		l2 = numpy.arange(10) + 10

		cls = MD.ClassSets(l1 = l1, l2 = l2)
		
		o, n, p = sample(cls)
		for i in xrange(len(o)) :
			if n[i] == 0. :
				self.assertEquals(o[i][1], 0.)
				self.assertEquals(o[i][0], 1.)
			else :
				self.assertEquals(o[i][0], 0.)
				self.assertEquals(o[i][1], 1.)

		nbTrials = 10000
		nb2 = 0.
		for i in xrange(nbTrials) :
			o, n, p = sample(cls)
			for j in xrange(len(p)) :
				if p[j] > 10 :
					nb2 += 1
		
		f = nb2/float(len(p)*nbTrials)
		r = abs(f-0.5)
		self.assertTrue(r < 2)

if __name__ == '__main__' :
	import Mariana.settings as MSET
	MSET.VERBOSE = False
	unittest.main()
