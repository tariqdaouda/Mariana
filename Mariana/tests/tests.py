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

def getMLP(self, nbInputs=2, nbClasses=2) :
    ls = MS.GradientDescent(lr = 0.1)
    cost = MC.NegativeLogLikelihood()

    i = ML.Input(nbInputs, 'inp')
    h = ML.Hidden(size = 6, activation = MA.ReLU(), name = "Hidden_0.500705866892")
    o = ML.SoftmaxClassifier(nbClasses=nbClasses, cost=cost, learningScenari=[ls], name = "out")

    mlp = i > h > o
    mlp.init()
    return mlp
    
class MLPTests(unittest.TestCase):

    def setUp(self) :
        numpy.random.seed(42)
        
        self.xor_ins = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]

        self.xor_outs = [0, 1, 1, 0]

    def tearDown(self) :
        pass

    def trainMLP_xor(self) :
        mlp = getMLP(2, 2)

        self.xor_ins = numpy.array(self.xor_ins)
        self.xor_outs = numpy.array(self.xor_outs)
        for i in xrange(1000) :
            mlp["out"].train({"inp.inputs": self.xor_ins, "out.targets" : self.xor_outs} )
        
        return mlp

    @unittest.skip("skipping")
    def test_missing_args(self) :
        mlp = getMLP(2, 2)
        self.assertRaises(SyntaxError, mlp["out"].train, {} )
    
    @unittest.skip("skipping")
    def test_unexpected_args(self) :
        mlp = getMLP(2, 2)
        self.assertRaises( SyntaxError, mlp["out"].train, {"inp.inputs": self.xor_ins, "out.targets" : self.xor_outs, "lala": 0} )

    @unittest.skip("skipping")
    def test_xor(self) :
        mlp = self.trainMLP_xor()

        pa = mlp["out"].accuracy["train"]({"inp.inputs": self.xor_ins, "out.targets" : self.xor_outs})["out.accuracy.train"]
        self.assertEqual(pa, 1)
        pc = mlp["out"].accuracy["test"]({"inp.inputs": self.xor_ins, "out.targets" : self.xor_outs})["out.accuracy.test"]
        self.assertEqual(pc, 1)
        
        for i in xrange(len(self.xor_ins)) :
            self.assertEqual(mlp["out"].predict["test"]( {"inp.inputs": [self.xor_ins[i]]} )["out.predict.test"], self.xor_outs[i] )
        
    @unittest.skip("skipping")
    def test_save_load_64h(self) :
        import os
        import Mariana.network as MN

        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.NegativeLogLikelihood()

        i = ML.Input(2, 'inp')
        o = ML.SoftmaxClassifier(nbClasses=2, cost=cost, learningScenari=[ls], name = "out")

        prev = i
        for i in xrange(64) :
            h = ML.Hidden(size=10, activation = MA.ReLU(), name = "Hidden_%s" %i)
            prev > h
            prev = h
        
        mlp = prev > o
        mlp.init()
        mlp.save("test_save")
        
        mlp2 = MN.loadModel("test_save.mar")
        mlp2.init()

        v1 = mlp["out"].propagate["test"]( {"inp.inputs": self.xor_ins} )["out.propagate.test"]
        v2 = mlp2["out"].propagate["test"]( {"inp.inputs": self.xor_ins} )["out.propagate.test"]
        self.assertTrue((v1==v2).all())
        os.remove('test_save.mar')

    @unittest.skip("skipping")
    def test_ae_reg(self) :
        powerOf2 = 3
        nbUnits = 2**powerOf2

        data = []
        for i in xrange(nbUnits) :
            zeros = numpy.zeros(nbUnits)
            zeros[i] = 1
            data.append(zeros)

        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.MeanSquaredError()

        i = ML.Input(nbUnits, name = 'inp')
        h = ML.Hidden(powerOf2, activation = MA.ReLU(), initializations=[MI.Uniform('W', small=True), MI.SingleValue('b', 0)], name = "hid")
        o = ML.Regression(nbUnits, activation = MA.ReLU(), initializations=[MI.Uniform('W', small=True), MI.SingleValue('b', 0)], learningScenari = [ls], cost = cost, name = "out" )

        ae = i > h > o
        ae.init()
        
        miniBatchSize = 1
        for e in xrange(2000) :
            for i in xrange(0, len(data), miniBatchSize) :
                miniBatch = data[i:i+miniBatchSize]
                ae["out"].train({"inp.inputs": miniBatch, "out.targets":miniBatch} )["out.drive.train"]
                
        res = ae["out"].propagate["test"]({"inp.inputs": data})["out.propagate.test"]
        for i in xrange(len(res)) :
            self.assertEqual( numpy.argmax(data[i]), numpy.argmax(res[i]))

    @unittest.skip("skipping")
    def test_ae(self) :
        powerOf2 = 3
        nbUnits = 2**powerOf2

        data = []
        for i in xrange(nbUnits) :
            zeros = numpy.zeros(nbUnits)
            zeros[i] = 1
            data.append(zeros)

        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.MeanSquaredError()

        i = ML.Input(nbUnits, name = 'inp')
        h = ML.Hidden(powerOf2, activation = MA.ReLU(), initializations=[MI.Uniform('W', small=True), MI.SingleValue('b', 0)], name = "hid")
        o = ML.Autoencode(targetLayer=i, activation = MA.ReLU(), initializations=[MI.Uniform('W', small=True), MI.SingleValue('b', 0)], learningScenari = [ls], cost = cost, name = "out" )

        ae = i > h > o
        ae.init()
        
        miniBatchSize = 1
        for e in xrange(2000) :
            for i in xrange(0, len(data), miniBatchSize) :
                miniBatch = data[i:i+miniBatchSize]
                loss = ae["out"].train({"inp.inputs": miniBatch} )["out.drive.train"]
                
        res = ae["out"].propagate["test"]({"inp.inputs": data})["out.propagate.test"]
        for i in xrange(len(res)) :
            self.assertEqual( numpy.argmax(data[i]), numpy.argmax(res[i]))
    
    @unittest.skip("skipping")
    def test_multiout_fctmixin(self) :
        
        i = ML.Input(1, name = 'inp')
        o1 = ML.Autoencode(targetLayer=i, activation = MA.Tanh(), learningScenari = [MS.GradientDescent(lr = 0.1)], cost = MC.MeanSquaredError(), name = "out1" )
        o2 = ML.Regression(1, activation = MA.Tanh(), learningScenari = [MS.GradientDescent(lr = 0.2)], cost = MC.MeanSquaredError(), name = "out2" )

        i  > o1
        ae = i > o2
        ae.init()
 
        preOut1 = ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"]
        preOut2 = ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"]
        ae["out1"].train({"inp.inputs": [[1]]} )["out1.drive.train"]
        self.assertTrue( preOut1 > ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"] )
        self.assertTrue( preOut2 == ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"] )

        preOut1 = ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"]
        preOut2 = ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"]
        ae["out2"].train({"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.train"]
        self.assertTrue( preOut1 == ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"] )
        self.assertTrue( preOut2 > ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"] )

        preOut1 = ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"]
        preOut2 = ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"]
        fct = ae["out1"].train + ae["out2"].train + ae["inp"].propagate["train"]
        ret = fct( {"inp.inputs": [[1]], "out2.targets": [[1]]} )
        self.assertEqual( len(ret), 3)
        self.assertTrue( preOut1 > ae["out1"].test( {"inp.inputs": [[1]]} )["out1.drive.test"] )
        self.assertTrue( preOut2 > ae["out2"].test( {"inp.inputs": [[1]], "out2.targets": [[1]]} )["out2.drive.test"] )
        
    @unittest.skip("skipping")
    def test_concatenation(self) :
        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.NegativeLogLikelihood()

        inp = ML.Input(2, 'inp')
        h1 = ML.Hidden(5, activation = MA.Tanh(), name = "h1")
        h2 = ML.Hidden(5, activation = MA.Tanh(), name = "h2")
        o = ML.SoftmaxClassifier(nbClasses=2, cost=cost, learningScenari=[ls], name = "out")
        
        inp > h1
        inp > h2
        c = ML.C([h1, h2], name="concat")
        mlp = c > o
        mlp.init()

        self.assertEqual( c.getIntrinsicShape()[0], h1.getIntrinsicShape()[0] + h2.getIntrinsicShape()[0])
        for i in xrange(10000) :
            ii = i%len(self.xor_ins)
            miniBatch = [ self.xor_ins[ ii ] ]
            targets = [ self.xor_outs[ ii ] ]
            mlp["out"].train({"inp.inputs": miniBatch, "out.targets":targets} )["out.drive.train"]

        for i in xrange(len(self.xor_ins)) :
            self.assertEqual(mlp["out"].predict["test"]( {"inp.inputs": [self.xor_ins[i]]} )["out.predict.test"], self.xor_outs[i] )

    @unittest.skip("skipping")
    def test_merge(self) :
        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.NegativeLogLikelihood()

        inp1 = ML.Input(1, 'inp1')
        inp2 = ML.Input(1, 'inp2')
        merge = ML.M((inp1 + inp2) / 3 * 10 -1, name = "merge")

        inp1 > merge
        mdl = inp2 > merge
        mdl.init()

        self.assertEqual( merge.getIntrinsicShape(), inp1.getIntrinsicShape())
        v = mdl["merge"].propagate["test"]({"inp1.inputs": [[1]],"inp2.inputs": [[8]]} )["merge.propagate.test"]
        self.assertEqual(v, 29)
    
    @unittest.skip("skipping")
    def test_embedding(self) :
        """the first 3 and the last 3 should be diametrically opposed"""
        data = [[0], [1], [2], [3], [4], [5]]
        targets = [0, 0, 0, 1, 1, 1]

        ls = MS.GradientDescent(lr = 0.5)
        cost = MC.NegativeLogLikelihood()
        
        inp = ML.Input(1, 'inp')
        emb = ML.Embedding(nbDimensions=2, dictSize=len(data), learningScenari = [ls], name="emb")
        o = ML.SoftmaxClassifier(2, learningScenari = [MS.Fixed()], cost = cost, name = "out")
        net = inp > emb > o
        net.init()

        miniBatchSize = 2
        for i in xrange(2000) :
            for i in xrange(0, len(data), miniBatchSize) :
                net["out"].train({"inp.inputs": data[i:i+miniBatchSize], "out.targets":targets[i:i+miniBatchSize]} )["out.drive.train"]

        embeddings = emb.getP("embeddings").getValue()
        for i in xrange(0, len(data)/2) :
            v = numpy.dot(embeddings[i], embeddings[i+len(data)/2])
            self.assertTrue(v < -1)

    @unittest.skip("skipping")
    def test_optimizer_override(self) :
        
        ls = MS.GradientDescent(lr = 0.5)
        cost = MC.NegativeLogLikelihood()
        
        inp = ML.Input(1, 'inp')
        h = ML.Hidden(5, activation = MA.Tanh(), learningScenari = [MS.Fixed("b")], name = "h")
        o = ML.SoftmaxClassifier(2, learningScenari = [MS.GradientDescent(lr = 0.5), MS.Fixed("W")], cost = cost, name = "out")
        net = inp > h > o
        net.init()

        ow = o.getP('W').getValue()
        ob = o.getP('b').getValue()
        hw = h.getP('W').getValue()
        hb = h.getP('b').getValue()
        for x in xrange(1,10):
            net["out"].train({"inp.inputs": [[1]], "out.targets":[1]} )["out.drive.train"]
        
        self.assertTrue(sum(ow[0]) == sum(o.getP('W').getValue()[0]) )
        self.assertTrue(sum(ob) != sum(o.getP('b').getValue()) )
        self.assertTrue(sum(hb) == sum(h.getP('b').getValue()) )
        self.assertTrue( sum(hw[0]) != sum( h.getP('W').getValue()[0]) )

    @unittest.skip("skipping")
    def test_conv_pooling(self) :
        import Mariana.convolution as MCONV
        import Mariana.sampling as MSAMP
        import theano

        def getModel(inpSize, filterWidth) :
            ls = MS.GradientDescent(lr = 0.5)
            cost = MC.NegativeLogLikelihood()

            i = ML.Input((1, 1, inpSize), name = 'inp')
            
            c1 = MCONV.Convolution2D( 
                numFilters = 5,
                filterHeight = 1,
                filterWidth = filterWidth,
                activation = MA.ReLU(),
                name = "conv1"
            )

            pool1 = MSAMP.MaxPooling2D(
                poolHeight = 1,
                poolWidth = 2,
                name="pool1"
            )

            c2 = MCONV.Convolution2D( 
                numFilters = 10,
                filterHeight = 1,
                filterWidth = filterWidth,
                activation = MA.ReLU(),
                name = "conv2"
            )

            pool2 = MSAMP.MaxPooling2D(
                poolHeight = 1,
                poolWidth = 2,
                name="pool1"
            )

            h = ML.Hidden(5, activation = MA.ReLU(), name = "hid" )
            o = ML.SoftmaxClassifier(nbClasses=2, cost=cost, learningScenari=[ls], name = "out")
            
            model = i > c1 > pool1 > c2 > pool2 > h > o
            model.init()
            return model

        def makeDataset(nbExamples, size, patternSize) :
            """dataset with pattern of 1 randomly inserted"""
            data = numpy.random.random((nbExamples, 1, 1, size)).astype(theano.config.floatX)
            data = data / numpy.sum(data)
            pattern = numpy.ones(patternSize)
            
            targets = []
            for i in xrange(len(data)) :
                if i%2 == 0 :
                    start = numpy.random.randint(0, size/2 - patternSize)
                    targets.append(0)
                else :
                    start = numpy.random.randint(size/2, size - patternSize)
                    targets.append(1)

                data[i][0][0][start:start+patternSize] = pattern

            targets = numpy.asarray(targets, dtype=theano.config.floatX)
            
            trainData, trainTargets = data, targets

            return (trainData, trainTargets)

        examples, targets = makeDataset(1000, 128, 6)
        model = getModel(128, 3)
        miniBatchSize = 32

        previousValues = {}
        for layer in ["conv1", "conv2"] :
            for param in ["W", "b"] :
                previousValues["%s.%s" % (layer, param)] = model[layer].getP(param).getValue().mean()

        for epoch in xrange(100) :
            for i in xrange(0, len(examples), miniBatchSize) :
                res = model["out"].train({"inp.inputs": examples[i:i+miniBatchSize], "out.targets":targets[i:i+miniBatchSize]} )["out.drive.train"]
        
        for layer in ["conv1", "conv2"] :
            for param in ["W", "b"] :
                self.assertFalse(previousValues["%s.%s" % (layer, param)] == model[layer].getP(param).getValue().mean() )

        self.assertTrue(res < 0.1)

    # @unittest.skip("skipping")
    def testRecurrences(self) :
        import Mariana.recurrence as MREC
        import Mariana.reshaping as MRES
        
        ls = MS.GradientDescent(lr = 0.1)
        cost = MC.NegativeLogLikelihood()
        
        # for clas in [MREC.RecurrentDense, MREC.LSTM, MREC.GRU] : 
        for clas in [MREC.RecurrentDense] : 
            inp = ML.Input((None, 3), 'inp')
            r = clas(2, onlyReturnFinal=True, name = "rec")
            reshape = MRES.Reshape((-1, 2), name="reshape")
            o = ML.SoftmaxClassifier(2, cost=cost, learningScenari = [MS.GradientDescent(lr = 0.5)], name = "out")
            net = inp > r > reshape > o
            net.init()
            inputs = [ [ [1, 1], [1, 2], [1, 1] ], [ [1, 1], [10, 1], [1, 10] ] ]
            
            oldWih = r.getP("W_in_to_hid").getValue()
            oldWhh = r.getP("W_hid_to_hid").getValue()
            for x in xrange(1,100):
                net["out"].train({"inp.inputs": inputs, "out.targets":[1, 1, 1]} )["out.drive.train"]
            self.assertTrue(oldWih.mean() != r.getP("W_in_to_hid").getValue().mean() )
            self.assertTrue(oldWhh.mean() != r.getP("W_hid_to_hid").getValue().mean() )
    
if __name__ == '__main__' :
    import Mariana.settings as MSET
    MSET.VERBOSE = False
    unittest.main()
