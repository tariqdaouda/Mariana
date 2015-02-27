
Mariana(Aplha)
==============

Named after the deepest place on earth (Mariana trench), Mariana is a machine learning framework on top of theano.
Mariana is still in active developement, but please feel free to play with it.

Why is it cool?
=========================

Most neural networks frameworks are way too complicated and cryptic. That really shouldn't be the case because whether your are building a Percetron an MLP or a Deep Auto-Encoder, your are basically doing the same thing: connecting layers and training them using backprob.

Mariana networks are graphs of independent layers and that allows for the craziest deepest architectures you can think of, and a very light and clear interface.
There's no need for an MLP or a Perceptron or an Auto-Encoder class, because if you know what these things are, you can turn one into the other in a few seconds.

Mariana is also completely agnostic regarding your datasets of the way you load your hyper-parameters. It's business is models and that's it.

So in short:
  
  * no YAML
  * no requirements concerning the format of the datasets
  * free you imagination and experiment

Can it run on GPU?
==================

At the heart of Mariana are theano functions, so the answer is yes.

Example
=======

.. code:: python
  
  #we define a learning scenario
  #you can have a per layer scenario, but if a hidden layer has no defined
  #scenario it inherits the scenario of the output
  ls = DefaultScenario(lr = 0.1, momentum = 0)
  #a cost
  cost = NegativeLogLikelihood(l1 = 0, l2 = 0)
  
  #we create our layers
  i = Input(2, 'inp')
  h = Hidden(4, activation = tt.tanh)
  o = SoftmaxClassifier(2, learningScenario = ls, costObject = cost, name = "out")
  #
  
  #this is our network ">" serves to connect layers together
  mlp = i > h > o
  preceptron = i > o
  
  #then it's simply a metter of calling
  #here "out" is the name of the output layer and "inp" the name of the input layer
  #Mariana networks can have several inputs/outputs. "target" is simply the target 
  mlp.train("out", inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ] )
  
You can also call mlp.test, mlp.propagate, mlp.classify. For more examples please have a look at the tests.

Saving and resuming training
============================

Models can be saved using the **save()** function:

.. code:: python

  mlp.save("myMLP")

Loading is a simple unpickling:

.. code:: python

  import cPickle
  
  mlp = cPickle.load(open("myMLP.mariana.pkl"))

Cloning layers
==============

Mariana allows you to clone layers so you can train a model, extract one of it's layers, and use in an other model.

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

Exentendable
============

Mariana allows you to define new types of layers, learning scenarios and costs by inheriting from the provided base
classes.
