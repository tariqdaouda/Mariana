Mariana: If you can draw it, you can make it.
=============================================

Named after the deepest place on earth (Mariana trench), Mariana is a Python Machine Learning Framework built on top of Theano, that focuses on ease of use. The full documentation is available here_.

.. _here: http://bioinfo.iric.ca/~daoudat/Mariana/

Creating Neural Networks with Mariana
=====================================

.. code:: python

	import Mariana.activations as MA
	import Mariana.decorators as MD
	import Mariana.layers as ML
	import Mariana.costs as MC
	import Mariana.regularizations as MR
	import Mariana.scenari as MS

**The instant MLP with dropout, L1 regularization and ReLUs**

.. code:: python

	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(28*28, name = "inputLayer")
	h = ML.Hidden(300, activation = MA.reLU, decorators = [MD.BinomialDropout(0.2)], regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])

	MLP = i > h > o
