
import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

"""A very basic auto encoder that learns to encode 8 bits into 3"""

miniBatchSize = 2

data = []
for i in xrange(8) :
	zeros = numpy.zeros(8)
	zeros[i] = 1
	data.append(zeros)

data = numpy.asarray(data)

ls = MS.GradientDescent(lr = 0.1)
cost = MC.MeanSquaredError()

i = ML.Input(8, name = 'inp')
h = ML.Hidden(3, activation = MA.reLU, name = "hid", saveOutputs = True )
o = ML.Regression(8, activation = MA.reLU, learningScenario = ls, costObject = cost, name = "out", saveOutputs = True )

ae = i > h > o

for e in xrange(1000) :
	for i in xrange(0, len(data), miniBatchSize) :
		#print data[i:i+miniBatchSize]
		ae.train(o, inp = data[i:i+miniBatchSize], targets = data[i:i+miniBatchSize] )

res = ae.propagate(o, inp = data)[0]
for i, r in enumerate(res) :
	m = numpy.max(r)
	t = []
	for rr in r :
		if rr == m :
			t.append(1)
		else :
			t.append(0)
	print t