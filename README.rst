
Mariana(Aplha)
==============

Named after the deepest place on earth (Mariana trench), Mariana is a machine learning framework on top of theano.
Mariana is still in active developement, but please feel free to play with it.

Flexibility
==============

Mariana networks are graphs of layers and that allows for 
the craziest deepest architectures you can think of.

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
  
  #then it's simply a metter of calling
  #here "out" is the name of the output layer and "inp" the name of the input layer
  #Mariana networks can have several inputs/outputs. "target" is simply the target 
  mlp.train("out", inp = [ self.xor_ins[ ii ] ], target = [ self.xor_outs[ ii ] ] )
  
You can also call mlp.test, mlp.propagate, mlp.classify. For more examples please have a look at the tests.
