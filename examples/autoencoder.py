import numpy

import Mariana.activations as MA
import Mariana.decorators as MD
import Mariana.layers as ML
import Mariana.costs as MC
import Mariana.regularizations as MR
import Mariana.scenari as MS

"""A very basic auto encoder that learns to encode 8 bits into 3"""

def makeData() :
	data = []
	for i in xrange(8) :
		zeros = numpy.zeros(8)
		zeros[i] = 1
		data.append(zeros)

	return data

def ae1(data) :
	'''Using a regression layer. This layer needs an explicit target'''

	miniBatchSize = 2

	ls = MS.GradientDescent(lr = 0.1)
	cost = MC.MeanSquaredError()

	i = ML.Input(8, name = 'inp')
	h = ML.Hidden(3, activation = MA.ReLU(), name = "hid")
	o = ML.Regression(8, activation = MA.ReLU(), learningScenario = ls, costObject = cost, name = "out")

	ae = i > h > o

	for e in xrange(1000) :
		for i in xrange(0, len(data), miniBatchSize) :
			ae.train(o, inp = data[i:i+miniBatchSize], targets = data[i:i+miniBatchSize] )

	return ae, o

def ae2(data) :
	"""This one uses an Autoencode layer. This layer is a part of the graph and does not need a specific traget"""
	
	miniBatchSize = 1

	ls = MS.GradientDescent(lr = 0.1)
	cost = MC.MeanSquaredError()

	i = ML.Input(8, name = 'inp')
	h = ML.Hidden(3, activation = MA.ReLU(), name = "hid")
	o = ML.Autoencode(i.name, activation = MA.ReLU(), learningScenario = ls, costObject = cost, name = "out")

	ae = i > h > o
	# ae.init()
	# o.train.printGraph()
	for e in xrange(1000) :
		for i in xrange(0, len(data), miniBatchSize) :
			ae.train(o, inp = data[i:i+miniBatchSize] )

	return ae, o

def printResults(ae, o, data) :
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

data = makeData()
ae, o = ae2(data)

ae.printLog()

printResults(ae, o, data)