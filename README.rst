What will happen now that Theano is no longer developed?
=========================================================

**Mariana works!** I still use it almost everyday.

I am still taking care of the maintenance and may still add some minor features. For the future, the most straightforward path would be a complete port to Tensorflow or PyTorch. Let me know if you'd like to help!

T .


.. image:: https://github.com/tariqdaouda/Mariana/blob/master/MarianaLogo.png
logo by  `Sawssan Kaddoura`_.

.. _Sawssan Kaddoura: http://sawssankaddoura.com

Mariana V1 is here_

.. _here: https://github.com/tariqdaouda/Mariana/tree/master

MARIANA: The Cutest Deep Learning Framework
=============================================
.. image:: https://img.shields.io/badge/python-2.7-blue.svg 
    
**TEMPORARY README FOR MARIANA V2.**

Mariana is meant to be a **efficient language** through which complex deep neural networks can be easily expressed and easily manipulated. It's simple enough for beginners and doesn't get much complicated. Intuitive, user-friendly and yet flexible enough for research. It's here to empower **researchers**, **teachers** and **students** alike, while greatly facilitating **AI knowledge transfer** into other domains.

.. code:: python

	import Mariana.layers as ML
	import Mariana.scenari as MS
	import Mariana.costs as MC
	import Mariana.activations as MA
	import Mariana.regularizations as MR

	import Mariana.settings as MSET

	ls = MS.GradientDescent(lr = 0.01, momentum=0.9)
	cost = MC.NegativeLogLikelihood()

	inp = ML.Input(28*28, name = "InputLayer")
	h1 = ML.Hidden(300, activation = MA.ReLU(), name = "Hidden1", regularizations = [ MR.L1(0.0001) ])
	h2 = ML.Hidden(300, activation = MA.ReLU(), name = "Hidden2", regularizations = [ MR.L1(0.0001) ])
	o = ML.SoftmaxClassifier(10, learningScenario = ls, costObject = cost, name = "Probabilities")

	#Connecting layers
	inp > h1 > h2
	concat = ML.C([inp, h2])

	MLP_skip = concat > o
	MLP_skip.init()
	
	#Visualizing
	MLP_skip.saveHTML("mySkipMLP")
    
    	#training:
	for i in xrange(1000) :
		MLP_skip["Probabilities"].train({"InputLayer.inputs": train_set[0], "Probabilities.targets": train_set[1]})
	
	#testing
		print MLP_skip["Probabilities"].test({"InputLayer.inputs": test_set[0], "Probabilities.targets": test_set[1]})
	
V2's most exciting stuff
=========================

V2 is almost a complete rewite of V1. It is much better.

What's done
-----------

* **New built-in visualization**: Interactive visuallization that shows architecture along with parameters, hyper-parameters as well as user defined notes. A great tool for collaboration.
* **Function mixins (my favorite)**: Mariana functions can now be added together! The result is a function that performs the actions of all its components at once. Let's say we have two output layers and want a function that optimises on losses for both outputs. Creating it is as simple as: f = out1.train + out2.train, and then calling f. Mariana will derive f from both functions adding the costs, calculating gradients and updates seemlessly in the background
* **Easy access to gradients and updates**: Just call .getGradients(), .getUpdates() on any function to get a view of either gradients of updates for all parameters.
* **User friendly error messages**: Functions will tell you what arguments they expect.
* **Very very clean code with Streams!**: You probably heard of batchnorm... and how it has a different behaviour in training and in testing. Well that simple fact can be the cause of some very messy DL code. With streams all this is over. Streams are parallel universes of execution for functions. You can define your own streams and have as many as you want. For batchnorm it mean that depending on the stream you call your function in (test or train), the behaviour will be different, even though you only changed one word.
* **Chainable optimization rules**: As in the previous version, layers inherit their learning scenari from outputs, but have the possibility to redifine them. This is still true, but rules can now be chained. Here's how to define a layer with fixed bias: l = Dense( learningScenari=[GradientDescent(lr = 0.1), Fixed('b')]) 
* **Just in time function compilation**: All functions (including mixins) are only compiled if needed.
* **Lasagne compatible**: Every lasagne layer can be seemlessly imported and used into Mariana
* **Convolutions, Deconvolutions (Transpose convolution), all sorts of convolutions...**
* **Much easier to extend**: The (almost) complete rewrite made for a much more cleaner code that is much more easy to extend. It is now much simpler to create your own layers, decorators, etc... Function that you need to implement end with *_abs* and Mariana has whole new bunch of custom type that support streams.
* **New merge layer**: Need a layer that is a linear combination of other layers? The new MergeLayer is perfect for that newLayer = M(layer1 + (layers3 * layer3) + 4 )
* **New concatenation layer**: newLayer = C([Layer1, layer2])
* **Unlimited number of inputs per layer**: Each layer used to be limited to one. Now it is infinit
* **Abstractions are now divided into trainable (layers, decorators, activations) and untrainable (scenari, costs, initializations)**: All trainable abstractions can hold parameters and have untrainable abstractions applied to them. PReLU will finally join ReLU as an activation!
* Fancy ways to go downhill: **Adam, Adagrad**, ...

What's almost done
-------------------

* Inclusion of popular recurrences (LSTM, recurent layers, ...)

What's next
-----------

* Complete refactorisation of training encapsulation. Training encapsulation was the least popular aspect of Mariana so far. I will completely rewrite it to give it the same level of intuitiveness as the rest of the framework. The next iterration will be a huge improvement.
* Arbitrary recurrences in the graph
