Examples
=============

Full
-----

Please have a look at the examples_ folder on github_. You will find the following files:
	* mnist_mlp_: Implementation and training of an MLP (neural network with one hidden layer) that makes use of the full setup with dataset mappers, trainer, stop criteria and decorators.
	* vanilla_mnist_perceptron_mlp_: Implementations of both a Perceptron (no hidden layers) and ans MLP (one hidden layer) but trained without dataset mappers and trainers.
	* ae_: a very basic auto encoder that learns to encode 8 bits into 3.


.. _examples: https://github.com/tariqdaouda/Mariana/tree/master/Mariana/examples
.. _github: https://github.com/tariqdaouda/Mariana
.. _mnist_mlp: https://github.com/tariqdaouda/Mariana/tree/master/Mariana/examples/mnist_mlp.py
.. _vanilla_mnist_perceptron_mlp: https://github.com/tariqdaouda/Mariana/tree/master/Mariana/examples/vanilla_mnist_perceptron_mlp.py
.. _ae: https://github.com/tariqdaouda/Mariana/tree/master/Mariana/examples/ae.py

Snippets
-------------

These are the basics, importations first:

.. code:: python

	import Mariana.activations as MA
	import Mariana.decorators as MD
	import Mariana.layers as ML
	import Mariana.costs as MC
	import Mariana.regularizations as MR
	import Mariana.scenari as MS

**This is an MLP in Mariana, with dropout, L1 regularization and ReLUs**

.. code:: python

	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()

	i = ML.Input(28*28, name = "inputLayer")
	h = ML.Hidden(300, activation = MA.reLU, decorators = [MD.BinomialDropout(0.2)], regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])

	MLP = i > h > o

**This is an autoencoder with tied weights**

.. code:: python

	ls = MS.GradientDescent(lr = 0.001)
	cost = MC.MeanSquaredError()

	i = ML.Input(10, name = "inputLayer")
	h = ML.Hidden(2, activation = MA.tanh, decorators = [ MD.GlorotTanhInit() ])
	o = ML.Regression(10, activation = MA.tanh, costObject = cost, learningScenario = ls)

	ae = i > h > o

	#tied weights, we need to force the initialisation of the weight first
	ae.init()
	o.W = h.W.T

Training, Testing and Propagating without a trainer:

.. code:: python

	#train the model for output 'o' function will update parameters and return the current cost
	print MLP.train(o, inputLayer = train_set[0][i : i +miniBatchSize], targets = train_set[1][i : i +miniBatchSize] )

	#the same as train but does not updated the parameters
	print MLP.test(o, inputLayer = test_set[0][i : i +miniBatchSize], targets = test_set[1][i : i +miniBatchSize] )

	#the propagate will return the output for the output layer 'o'
	print MLP.propagate(o, inputLayer = test_set[0][i : i +miniBatchSize])

That's it for the snippets. There's also a trainer that make things even easier: `Training, Testing and Driving Nets in Mariana`_

.. _`Training, Testing and Driving Nets in Mariana`: training.html#trainers
