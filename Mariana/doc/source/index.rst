.. Mariana documentation master file, created by
   sphinx-quickstart on Thu Jun 11 13:02:52 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mariana: Deep Neural Networks should be Easy to Write
=====================================================

Named after the deepest place on earth (Mariana trench), Mariana is a Python Machine Learning Framework built on top of Theano_, that focuses on ease of use.  Mariana lives on github_.

.. _Theano: http://www.deeplearning.net/software/theano/
.. _github: https://github.com/tariqdaouda/Mariana/

Why is it cool?
----------------

**If you can draw it, you can write it.**

Mariana provides an interface so simple and intuitive that writing models becomes a breeze.
Networks are graphs of connected layers and that allows for the craziest deepest architectures 
you can think of, as well as for a super light and clean interface. The paradigm is simple
create layers and connect them using **'>'**. Plugging per layer regularizations, costs and things
such as dropout and custom initilisations is also super easy.

Here's a snippet for an instant MLP with dropout, ReLU units and L1 regularization:

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
  h = ML.Hidden(300, activation = MA.ReLU(), decorators = [MD.BinomialDropout(0.2)], regularizations = [ MR.L1(0.0001) ])
  o = ML.SoftmaxClassifier(9, learningScenario = ls, costObject = cost, regularizations = [ MR.L1(0.0001) ])
  
  MLP = i > h > o

Here are some full fledged examples_.

Mariana also supports trainers_ that encapsulate the whole training to make things even easier.

So in short:
  
  * no YAML
  * completely modular and extendable
  * use the trainer to encapsulate your training in a safe environement
  * write your models super fast
  * save your models and resume training
  * export your models into DOT format to obtain clean and easy to communicate graphs
  * free your imagination and experiment
  * no requirements concerning the format of the datasets


.. _trainers: training.html#trainers
.. _examples: intro.html


Extendable
----------

Mariana is an extendable framework that provides abstract classes allowing you to define new types of layers, learning scenarios, costs, stop criteria, recorders and trainers. Feel free to taylor it to your needs.

Contents:

.. toctree::
   :maxdepth: 2

   installation
   intro
   training
   gpu
   layers
   convolution
   network
   costs
   scenari
   activation
   decorators
   regularisation
   wrappers
   candies
   tips

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

