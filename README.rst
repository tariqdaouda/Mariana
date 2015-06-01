
Mariana(Alpha)
==============

Named after the deepest place on earth (Mariana trench), Mariana is a machine learning framework on top of theano.
Mariana is still in active developement, and there might be some bugs lurking in the dark. But I use it everyday, and bugs
tend to be rapidely corrected. Please feel free to play with it.

Why is it cool?
===============

Mariana networks are graphs of independent layers and that allows for the craziest deepest architectures 
you can think of, as well as for a super light and clear interface.
There's no need for an MLP or a Perceptron or an Auto-Encoder class,
because if you know what these things are, you can turn one into the other in a few seconds.

And that's the main objective behind Mariana, provide an interface so simple and intuitive that writing models
becomes a breeze.

Mariana is also completely agnostic regarding your datasets or the way you load your hyper-parameters. It's business is models and that's it.

So in short:
  
  * no YAML
  * write your models super fast
  * save your models and resume training
  * export your models to DOT format to obtain clean and easy to communicate graphs
  * free your imagination and experiment
  * no requirements concerning the format of the datasets

A word about the **'>'**
======================

When communicating about neural networks people often draw sets of connected layers. That's the idea behind Mariana: layers are first defined, then connected using the **'>'** operator. 

Short examples
===============

Importations first

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
	
	i = ML.Input(28*28)
	h = ML.Hidden(300, activation = MA.reLU, decorators = [MD.BinomialTurnOff(0.2)], regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])
	
	MLP = i > h > o

**This is an autoencoder with tied weights**

.. code:: python

	ls = MS.GradientDescent(lr = 0.001)
	cost = MC.MeanSquaredError()
	
	i = ML.Input(10)
	h = ML.Hidden(2, activation = MA.tanh, decorators = [ MD.GlorotTanhInit() ])
	o = ML.Regression(10, activation = MA.tanh, costObject = cost, learningScenario = ls)
	
	ae = i > h > o
	ae.init()
	
	#tied weights
	o.W = h.W.T

Can it run on GPU?
==================

At the heart of Mariana are theano functions, so the answer is yes. The guys behind theano really did an awesome
job of optimization, so it should be pretty fast, wether you're running on CPU or GPU.

Example
=======

Please have a look at **mnist_mlp.py** in the examples folder. It illustrates most of what this quickstart guide adresses.
There's also **vanilla_mnist_perceptron_mlp.py**, wich demonstrate how to train an MLP (network with one hidden layer) or a Percetron on mnist
without the use of a trainer.

Using the trainer and loading datasets
========================================

Trainers and Recorders
----------------------

The trainer takes care of the whole training process. If the process dies unexpectedly during training it will also automatically save the last version of the model as well as logs explaining what happened. The trainer can also take as argument a list of stopCriterias, and be
paired with a recorder whose job is to record the training evolution.
For now there is only one recorder GGPlot2 (which is also the default recorder).

This recorder will:

	* Output the training results for each epoch, highliting every time a new best score is achieved
	* Automatically save the model each time a new best score is achieved
	* Create and update a *CSV file* in a GGPlot2 friendly format that contains the whole historic of the training as well as information such as runtime and hyperparameter values.

Dataset maps
------------

Mariana is dataset format agnostic. The way it works is that you map sets to specific input and output layers, cf. the mnist example.

Decorators
==========

Mariana layers can take decarators as arguments that modify the layer's behaviour. Decorators can be used for example, to mask parts of the output to the next layers (ex: for dropout or denoising auto-encoders),
or to specify custom weight initialisations.

Costs and regularizations
=========================

Each output layers can have its own cost. Regularizations are also specified on per layer basis, so you can for example enforce a L1 regularisation on a single layer of the model.

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

Getting the outputs of intermediate layers
==========================================

By setting a layer with the argument **saveOutputs=True**. You tell Mariana to keep the last ouputs of that layer stored, so you can access them using **.getLastOutputs()** function.

Visualizing networks
====================

To get a DOT format representation of your network:

.. code:: python
  
  #to simply print it
  print mlp.toDOT()

  #to save it
  mlp.saveDOT("myMLP.dot")

You can then visualize your graph with any DOT visualizer such a graphviz.

Extendable
============

Mariana allows you to define new types of layers, learning scenarios, costs, stop criteria, recorders and trainers by inheriting from the provided base classes. Feel free to taylor it to your needs.
