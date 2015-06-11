
Training, Testing and Driving Nets in Mariana
=============================================
All the concepts introduced here are demonstrasted in the `Examples`_. section.

.. _Examples:

Vanilla Driving
---------------

Models in Mariana can be driven output layer by output layer using the model functions *train*, *test* and *propagate*

.. code:: python

	import Mariana.activations as MA
	import Mariana.decorators as MD
	import Mariana.layers as ML
	import Mariana.costs as MC
	import Mariana.regularizations as MR
	import Mariana.scenari as MS

	ls = MS.GradientDescent(lr = 0.01)
	cost = MC.NegativeLogLikelihood()
	
	i = ML.Input(28*28, name = "inputLayer")
	h = ML.Hidden(300, activation = MA.reLU, decorators = [MD.BinomialDropout(0.2)], regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])
	
	MLP = i > h > o
	
	...load you dataset etc...

	#train the model for output 'o' function will update parameters and return the current cost
	print MLP.train(o, inputLayer = train_set[0][i : i +miniBatchSize], target = train_set[1][i : i +miniBatchSize] )

	#the same as train but does not updated the parameters
	print MLP.test(o, inputLayer = test_set[0][i : i +miniBatchSize], target = test_set[1][i : i +miniBatchSize] )
	
	#the propagate will return the output for the output layer 'o'
	print MLP.propagate(o, inputLayer = test_set[0][i : i +miniBatchSize])

Trainers
--------

A trainer takes care of the whole training process. If the process dies unexpectedly during training it will 
automatically save the last version of the model as well as logs explaining what happened. The trainer can also take as argument a list of stopCriterias, and be paired with a recorder whose job is to record the training evolution. Trainers must has a .store dictionary exposed that represents the current state of the process. The store is used by stop criteria, recorders and learning scenarii.

.. automodule:: Mariana.training.trainers
   :members:

Dataset maps
------------

Mariana is dataset format agnostic and uses **DatasetMaps** to associate input and output layers with the data they must receive, cf. **examples/mnist_mlp.py** for an example. Here's a short example of how it works::

	i = ML.Input(...)
	o = ML.Output(...)

	trainSet = RandomSeries(images = train_set[0], classes = train_set[1])
	
	DatasetMapper = dm
	dm.map(i, trainSet.images)
	dm.map(o, trainSet.classes)

.. automodule:: Mariana.training.datasetmaps
   :members:

Stop criteria
--------------

Stop criteria are simply there to tell to observe the trainer's store and tell it when it should stop by raising an EndOfTraining exception.

.. automodule:: Mariana.training.stopcriteria
   :members:

Recorders
---------

Recorders are objetct meant to be plugged into trainers to record the advencement of the training. They take the .store of a trainer and
do something smart and useful with it. As with everythin in Mariana feel free to write your own recoders.

.. automodule:: Mariana.training.recorders
   :members: