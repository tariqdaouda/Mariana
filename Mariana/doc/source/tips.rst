Tips
===============

Less Verbosity
---------------

.. code:: python

 	import Mariana.settings as MSET

 	MSET.VERBOSE = False

Changing the seed of random generators
---------------------------------------

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

You can then visualize your graph with any DOT visualizer such a graphviz.