Tips and FAQs
===============


What can you do with Mariana
----------------------------

Any type of deep, shallow, feed-forward, back-prop trained Neural Network should work. Convolutional Nets and Recurrent Nets are not supported yet but they will be.


A word about the **'>'**
-------------------------

When communicating about neural networks people often draw sets of connected layers. That's the idea behind Mariana: layers are first defined, then connected using the **'>'** operator. 

Can it run on GPU?
------------------

At the heart of Mariana are Theano functions, so the answer is yes. The guys behind Theano really did an awesome
job of optimization, so it should be pretty fast, wether you're running on CPU or GPU.


Less Verbosity
---------------

.. code:: python

 	import Mariana.settings as MSET

 	MSET.VERBOSE = False

Modifiying hyper-parameters during training
--------------------------------------------

If you are not using a trainer you can simply change the values of the hyper-parameters of the learning scenario inside your loop.

If you are using trainer, learning scenarii have an **update(self, trainer)** function that is called by the trainer at each epoch. Trainers have a **.store** dictionary attribute that stores values relative to the current epoch (for example the current epoch number is contained in **trainer.store["runInfos"]["epoch"]**). The role of this function is to modidy the attribute of the learning scenario according to the values in the store.
You may need to create your own learning scenario, for that, simply write a class that inherits from an existing learning scenario or from the provided base class.

Getting the outputs of intermediate layers
-------------------------------------------

By initialising a layer with the argument::

  saveOutputs=True

You tell Mariana to keep the last outputs of that layer stored, you can then access them using the layer's "getLastOutputs()"" function.

Changing the seed random parameters generation
----------------------------------------------

.. code:: python

 	import Mariana.settings as MSET

 	MSET.RANDOM_SEED = 5826

Saving and resuming training
-----------------------------

Models can be saved using the **save()** function:

.. code:: python

  mlp.save("myMLP")

Loading is a simple unpickling:

.. code:: python

  import cPickle
  
  mlp = cPickle.load(open("myMLP.mariana.pkl"))
  mlp.train(...)

Cloning layers and re-using layers
-----------------------------------

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
---------------------

Networks can be exported to graphs in the DOT format:

.. code:: python
  
  #to simply print it
  print mlp.toDOT()

  #to save it
  mlp.saveDOT("myMLP.dot")

You can then visualize the graph with any DOT visualizer such a graphviz.