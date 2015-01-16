
 > Mariana > (pre-aplha):
========================

Named after the deepest place on earth (Marianas trench), Mariana is a machine learning framework on top of theano.

What to expect:
==============

The one liner MLP

.. code:: python

  mlp = Input(10) > Hidden(4) > Classifier(2)
  
  for x, y in trainingSet.iteritems() :
    mlp.train(x, y)
