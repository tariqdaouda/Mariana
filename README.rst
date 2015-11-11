
Mariana
==============

Named after the deepest place on earth (Mariana trench), Mariana is a Python Machine Learning Framework built on top of Theano, that focuses on ease of use. The full documentation is available here_.

.. _here: http://bioinfo.iric.ca/~daoudat/Mariana/

Why is it cool?
===============

**If you can draw it, you can write it.**

Mariana provides an interface so simple and intuitive that writing models becomes a breeze.
Networks are graphs of connected layers and that allows for the craziest deepest architectures
you can think of, as well as for a super light and clean interface.

There's no need for an MLP or a Perceptron or an Auto-Encoder class,
because if you know what these things are, you can turn one into the other in 2 seconds.

So in short:

* no YAML
* very easy to use
* great for Feed Forward nets (MLPs, Auto-Encoders, ...)
* supports momentum
* completely modular and extendable
* trainers can be used to encapsulate your training in a safe environement
* oversampling is also taken care of
* easily save your models and resume training
* export you models into HTML or DOT for easy communication and debuging
* free your imagination and experiment
* no requirements concerning the format of the datasets

A word about the **'>'**
======================

When communicating about neural networks people often draw sets of connected layers. That's the idea behind Mariana: layers are first defined, then connected using the **'>'** operator.

Installation
=============
First, make sure you have the latest version of Theano_ (do a git clone not a pip install).

Then Clone it from git!::

	git clone https://github.com/tariqdaouda/Mariana.git
	cd Mariana
	python setup.py develop

Update::

	git pull #from Mariana's folder
	
.. _Theano: https://github.com/Theano/Theano


Important notice
-----------------

If you run into a problem please try to update Theano first by doing a **git pull** in theano's folder.

Full Examples
=============

Please have a look at **examples/mnist_mlp.py**. It illustrates most of what this quickstart guide adresses.
There's also **examples/vanilla_mnist_perceptron_mlp.py**, wich demonstrate how to train an MLP (network with one hidden layer) or a Perceptron on the MNIST database
without the use of a trainer.

Short Snippets
===============

Importations first

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

Training, Testing and Propagating:

.. code:: python

	#train the model for output 'o' function will update parameters and return the current cost
	print MLP.train(o, inputLayer = train_set[0][i : i +miniBatchSize], target = train_set[1][i : i +miniBatchSize] )

	#the same as train but does not updated the parameters
	print MLP.test(o, inputLayer = test_set[0][i : i +miniBatchSize], target = test_set[1][i : i +miniBatchSize] )

	#the propagate will return the output for the output layer 'o'
	print MLP.propagate(o, inputLayer = test_set[0][i : i +miniBatchSize])

**This is an autoencoder with tied weights**

.. code:: python

	ls = MS.GradientDescent(lr = 0.001)
	cost = MC.MeanSquaredError()

	i = ML.Input(10, name = "inputLayer")
	h = ML.Hidden(2, activation = MA.tanh, decorators = [ MD.GlorotTanhInit() ])
	o = ML.Regression(10, activation = MA.tanh, costObject = cost, learningScenario = ls)

	ae = i > h > o
	ae.init()

	#tied weights, we need to force the initialisation of the weight first
	ae.init()
	o.W = h.W.T

Another way is to use the Autoencode layer as output::

	o = ML.Autoencode(i, activation = MA.tanh, costObject = cost, learningScenario = ls)

Can it run on GPU?
==================

At the heart of Mariana are Theano functions, so the answer is yes. The guys behind Theano really did an awesome
job of optimization, so it should be pretty fast, whether you're running on CPU or GPU.

Making life even easier: Trainers and Recorders
===============================================

A trainer takes care of the whole training process. If the process dies unexpectedly during training it will also automatically save the last version of the model as well as logs explaining what happened. The trainer can also take as argument a list of stopCriterias, and be
paired with a recorder whose job is to record the training evolution.
For now there is only one recorder : GGPlot2 (default recorder).

This recorder will:

* Output the training results for each epoch, highliting every time a new best score is achieved
* Automatically save the model each time a new best score is achieved
* Create and update a *CSV file* in a GGPlot2 friendly format that contains the entire history of the training as well as information such as runtime and hyperparameter values.

Dataset maps
------------

Mariana is dataset format agnostic and uses **DatasetMaps** to associate layers with the data the must receive, cf. **examples/mnist_mlp.py** for an example.

Decorators
==========

Mariana layers can take decorators as arguments that modify the layer's behaviour. Decorators can be used for example, to mask parts of the output to the next layers (ex: for dropout or denoising auto-encoders),
or to specify custom weight initializations.

Costs and regularizations
=========================

Each output layers can have its own cost. Regularizations are also specified on a per layer basis, so you can for example enforce a L1 regularisation on a single layer of the model.

Saving and resuming training
============================

Models can be saved using the **save()** function:

.. code:: python

  mlp.save("myMLP")

Loading is a simple unpickling:

.. code:: python

  import cPickle

  mlp = cPickle.load(open("myMLP.mariana.pkl"))
  mlp.train(...)

Getting the outputs of intermediate layers
==========================================

By setting a layer with the argument **saveOutputs=True**. You tell Mariana to keep the last outputs of that layer stored, so you can access them using **.getLastOutputs()** function.

Cloning layers and re-using layers
===================================

Mariana allows you to clone layers so you can train a model, extract one of it's layers, and use it for another model.

.. code:: python

  h2 = h.clone()

You can also transform an output layer into a hidden layer, that you can include afterwards in an other model.

.. code:: python

  h3 = o.toHidden()

And a hidden layer to an output layer using:

.. code:: python

  o = h.toOutput(ML.Regression, costObject = cost, learningScenario = ls)

Visualizing networks
====================

To simplify debugging and communication Mariana allow to export graphical representation of networks.

The easiest way is to export it as a web page:

.. code:: python

  #to save it
  mlp.saveHTML("myAwesomeMLP")

But you can also ask for a DOT format representation of your network:

.. code:: python

  #to simply print it
  print mlp.toDOT()

  #to save it
  mlp.saveDOT("myAwesomeMLP")

You can then visualize your graph with any DOT visualizer such a graphviz.

Extendable
============

Mariana allows you to define new types of layers, learning scenarios, costs, stop criteria, recorders and trainers by inheriting from the provided base classes. Feel free to taylor it to your needs.
