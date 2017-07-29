import unittest

import Mariana.layers as ML
import Mariana.initializations as MI
import Mariana.decorators as MD
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS
import Mariana.activations as MA

import theano.tensor as tt
import numpy

class DecoratorTests(unittest.TestCase):

    def setUp(self) :
        pass

    def tearDown(self) :
        pass

    # @unittest.skip("skipping")
    def test_batch_norm(self) :
        import theano, numpy
        
        def batchnorm(W, b, data) :
            return numpy.asarray( W * ( (data-numpy.mean(data)) / numpy.std(data) ) + b, dtype= theano.config.floatX)

        data = numpy.random.randn(1, 100).astype(theano.config.floatX)
        batch = MD.BatchNormalization(testMu=0, testSigma=1)
        inp = ML.Input(100, name='inp', decorators=[batch])
        
        model = inp.network
        model.init()

        m1 = numpy.mean( model["inp"].propagate["train"]({"inp.inputs": data})["inp.propagate.train"] )
        m2 = numpy.mean( batchnorm(batch.getP("gamma").getValue(), batch.getP("beta").getValue(), data) )

        epsilon = 1e-6
        self.assertTrue ( (m1 - m2) < epsilon )
    
    # @unittest.skip("skipping")
    def test_mask(self) :
        import theano, numpy
        
        inp = ML.Input(100, 'inp', decorators=[MD.Mask(mask = numpy.zeros(100))])
        model = inp.network
        model.init()
        
        data = numpy.random.randn(1, 100).astype(theano.config.floatX)
        out = model["inp"].propagate["train"]({"inp.inputs": data})["inp.propagate.train"]
        
        self.assertEqual(sum(out[0]), 0)

if __name__ == '__main__' :
    import Mariana.settings as MSET
    MSET.VERBOSE = False
    unittest.main()
