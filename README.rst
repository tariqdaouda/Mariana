
Mariana(Beta)
==============

Named after the deepest place on earth (Mariana trench), Mariana is a machine learning framework on top of theano.
Mariana is still in active developement, but please feel free to play with it.

Why is it cool?
=========================

Mariana networks are graphs of independent layers and that allows for the craziest deepest architectures you can think of, 
as well as for a super light and clear interface.
There's no need for an MLP or a Perceptron or an Auto-Encoder class, because if you know what these things are, you can turn one into the other in a few seconds.

Mariana is also completely agnostic regarding your datasets or the way you load your hyper-parameters. It's business is models and that's it.

So in short:
  
  * no YAML
  * write your models super fast
  * save your models and resume training
  * export your models to DOT format to obtain clean and easy to communicate graphs
  * free your imagination and experiment
  * no requirements concerning the format of the datasets
  
Compeling examples of cuteness
==============================
Importations first

.. code:: python

	import Mariana.activations as MA
	import Mariana.decorators as MD
	import Mariana.layers as ML
	import Mariana.costs as MC
	import Mariana.regularizations as MR
	import Mariana.scenari as MS

This is an MLP in Mariana with dropout, L1 regularization and ReLUs
---------------------------------------------------------------------

.. code:: python

	ls = MS.DefaultScenario(lr = 0.01, momentum = 0)
	cost = MC.NegativeLogLikelihood()
	
	i = ML.Input(28*28)
	h = ML.Hidden(300, activation = MA.reLU, decorators = [MD.StochasticTurnOff(0.2)], regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])
	
	MLP = i > h > o

This is an autoencoder with tied weights
----------------------------------------

.. code:: python

	ls = MS.DefaultScenario(lr = 0.001, momentum = 0.009)
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

Please have a look mnist_mlp.py in the examples folder. It illustrates most of what this quickstart guide adresses.

Using the trainer and loading datasets
========================================

Trainer
--------

The trainer takes care of the whole training process. If any exception occurs during training it will also automatically save the last
version of the model as well as logs explaining what happened. The trainer can also take as argument a list of stopCriterias.

It will:

	* Output the training results for each epoch, highliting every time a new best test error is achieved
	* Automatically save the model each time a new best test error is achieved
	* Create and update a *CSV file* that contains the whole historic of the training as well as information such as the hyperparameters. You can later compile several of those files, and plot for example the test error with respect to the number of hidden units

Dataset maps
------------

Mariana is dataset format agnostic. In order to use your dataset you will need to define maps for the differents sets that you need.

Let's assume that our sets are in a python dictionary such as:

.. code:: python

	sets =  {
			"set1" : {
				"images" : [....],
				"classes" : [....]
				},
			"set2" : {
				"images" : [....],
				"classes" : [....]
				}
			}

Using *DatasetMappers* we can now specify wich sets to use for training 
and testing for each input and each output of our model.
*Mariana networks can have multiple inputs and outputs, but here we only have a 
neural network with one input and one output.*

.. code:: python

	#here we decide that we are going to use "set1" as the training set and we map the input layer
	#to the "images" list of "set1", and the output layer to the "classes" list of the same set.
	trainMaps = tra.DatasetMapper()
	trainMaps.addInput("the input", sets["set1"]["images"])
	trainMaps.addOutput("the output", sets["set1"]["classes"])

	#we do the same with "set2", that we plan to use as our test set
	testMaps = tra.DatasetMapper()
	testMaps.addInput("the input", sets["set2"]["images"])
	testMaps.addOutput("the output", sets["set2"]["classes"])

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
  
Cloning layers and resusing layers
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

To get a DOT format representation of your network:

.. code:: python
  
  #to simply print it
  print mlp.toDOT()

  #to save it
  mlp.saveDOT("myMLP.dot")

You can then visualize your graph with any DOT visualizer such a graphviz.

Extendable
============

Mariana allows you to define new types of layers, learning scenarios and costs by inheriting from the provided base
classes.
