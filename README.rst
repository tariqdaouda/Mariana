
Mariana(Aplha)
==============

Named after the deepest place on earth (Mariana trench), Mariana is a machine learning framework on top of theano.
Mariana is still in active developement, but please feel free to play with it.

Why is it cool?
=========================

Mariana networks are graphs of independent layers and that allows for the craziest deepest architectures you can think of, and a very light and clear interface.
There's no need for an MLP or a Perceptron or an Auto-Encoder class, because if you know what these things are, you can turn one into the other in a few seconds.

Mariana is also completely agnostic regarding your datasets or the way you load your hyper-parameters. It's business is models and that's it.

So in short:
  
  * no YAML
  * no requirements concerning the format of the datasets
  * save your models and resume training
  * export your models to DOT format to obtain clean and easy to communicate graphs
  * free your imagination and experiment

Can it run on GPU?
==================

At the heart of Mariana are theano functions, so the answer is yes.

Example
=======

.. code:: python
  
  import Mariana.layers as lay
  import Mariana.rules as rul
	
  #we define a learning scenario
  #you can have a per layer scenario, but if a hidden layer has no defined
  #scenario it inherits the scenario of the output
  ls = rul.DefaultScenario(lr = 0.1, momentum = 0)
  #a cost
  cost = rul.NegativeLogLikelihood(l1 = 0, l2 = 0)
  
  #we create our layers
  i = lay.Input(2, 'inp')
  h = lay.Hidden(4, activation = tt.tanh)
  o = lay.SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")
  
  #this is our network ">" serves to connect layers together
  mlp = i > h > o
  preceptron = i > o
  
  #then it's simply a matter of calling
  #here "out" is the name of the output layer and "inp" the name of the input layer
  #Mariana networks can have several inputs/outputs. "target" is simply the target 
  mlp.train("out", inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ] )
  
You can also call mlp.test, mlp.propagate, mlp.classify. For more examples please have a look at the *tests* and *examples* folders.

Using Trainers and loading datasets
========================================

Trainers
--------

Trainers are objects that take care of the whole training process. If any exception occurs during training the trainer will also automatically save the last
version of the model as well as logs explaining what happened.

For now Mariana ships with only one trainer: **NoEarlyStopping**. This trainers takes a *test set* and a *train set* and will either run forever (nbEpochs = -1) or for a given number of epochs.

It will:

	* Output the training results for each epoch, highliting every time a new best test error is achieved
	* Automatically save the model each time a new best test error is achieved
	* Create and update a *CSV file* that contains the whole historic of the training as well as information such as the hyperparameters. You can later compile several of those files, and plot for example the test error with respect to the number of hidden units

The **trainers.py** module has a *Trainer* class that you can extend
to create custom trainers.

Dataset maps
------------

Mariana is dataset format agnostic. In order to use your dataset you will need to define maps for the differents sets that you need.

First let's create our model

.. code:: python

	import Mariana.layers as lay
	import Mariana.rules as rul
	import Mariana.trainers as tra

	#Let's define the network
	ls = rul.DefaultScenario(lr = 0.01, momentum = 0)
	cost = rul.NegativeLogLikelihood(l1 = 0, l2 = 0.0001)

	i = lay.Input(28*28, 'the input')
	h = lay.Hidden(500, activation = tt.tanh)
	o = lay.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "the output")

	mlp = i > h > o

Now let's assume that our sets are in a python dictionary such as:

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

	#We instanciate a trainer
	trainer = NoEarlyStopping()
	
	#and pass it the model as well as the maps.
	#nbEpochs = -1 means that the process will run forever until someone kills it
	trainer.run("Awesome MLP", 
		mlp, 
		trainMaps = trainMaps, 
		testMaps = testMaps, 
		nbEpochs = -1, 
		miniBatchSize = 20)
	
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
  
Cloning layers
==============

Mariana allows you to clone layers so you can train a model, extract one of it's layers, and use it for another model.

.. code:: python

  h2 = h.clone()

You can also transform an output layer into a hidden layer, that you can include afterwards in an other model.

.. code:: python

  h3 = o.toHidden()

Visualizing graphs
==================

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
