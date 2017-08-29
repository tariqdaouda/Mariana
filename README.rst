.. image:: https://travis-ci.org/tariqdaouda/Mariana.svg
    :target: https://travis-ci.org/tariqdaouda/Mariana.svg?branch=V2-dev
.. image:: https://codecov.io/gh/tariqdaouda/Mariana/branch/V2-dev/graph/badge.svg
    :target: https://codecov.io/gh/tariqdaouda/Mariana/branch/V2-dev/graph/
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://opensource.org/licenses/Apache-2.0

.. image:: https://img.shields.io/badge/python-2.7-blue.svg 

.. image:: https://avatars2.githubusercontent.com/u/7224902?v=3&s=460 

MARIANA: The Cutest Deep Learning Framework
=============================================

**TEMPORARY README FOR MARIANA V2, WIP (BUT ALMOST DONE).**

Mariana is an **efficient language** through which complex deep neural networks can be easily expressed and easily manipulated. It's simple enough for beginners and doesn't get much complicated for seniors. Intuitive, user-friendly and yet flexible enough for research. It's here to empower **researchers**, **teachers** and alike **students**, while greatly facilitating **AI knowledge transfer** into other domains.

V2's most exciting stuff
=========================

V2 is almost a complete rewite of V1. It is much better.

What's done
-----------

* **Convolutions, Deconvolutions (Transpose convolution)**

* **Lasagne compatible**: Every lasagne layer can be seemlessly imported and used in Mariana
* **Just in time function compilation**: At the heart of Mariana are Theano function. The previous version compiled every function a model initialization. This caused compilation times to be longer that needed. With this version functions are compiled only if needed.
* **Function mixins (My favorite)**: Mariana functions can now be added together! The result is a function that performs the action all its components. Let's say we have two output layers on want a function that optimises on losses for both outputs. Creating it is as simple as: f = out1.train + out2.train, and then calling f. Mariana will derive f from both functions adding the costs, calculating gradients and updates seemlessly in the background  
* **Streams!**: You probably heard of batchnorm... and how it has a different behaviour in training in testing. Well that simple fact can be the cause for some very messy DL code. With streams it is over. Streams are parrelle universes of execution for functions. You can define your own and have as many as you want. For batchnorm it mean that depending on the stream you call your function in (test or train), the behaviour will be different, but you only changed one word.
* **Much easier to extend**: The (almost) complete rewrite made for much more cleaner that is much more easy to extend. It is now much simpler to create your own layers, decorators, etc... Function that you need to implement end with *_abs* and Mariana has whole new bunch of custom type that support streams.
* **New merge layer**: Need a layer that is a linear combination of other layer layer. The new MergeLayer is perfect for that newLayer = M(layer1 + (layers3 * layer3) + 4 )
* **New concatenation layer**: newLayer = C([Layer1, layer2])
* **Unlimited number of inputs per layer**: Each layer used to be limited to one. Now it is infinit
* **Abstractions are now divided into trainable (layers, decorators, activations) and untrainable (scenari, costs, initializations)**: All trainable abstractions can hold parameters and have untrainable abstractions applied to them. PReLU will finally join ReLU as an activation!


What's almost done
-------------------

* Inclusion of popular recurrences (LSTM, recurent layers)
* New built-in visualisaton: The previous visualization only showed the architecture and layer shapes. The new one will be interactive. It will contain information on all parameters, hyper-parameters as well as user defined notes on the model, layers, or any other abstraction in thhe network. A great tool for collaboration.

What's next
-----------

* Complete refactorisation of training encapsulation. Training encapsulation was the least popular aspect of Mariana so far. I will completely rewrite it to give it the same level of intuitiveness as the rest of the framework. The next iterration will be a huge improvement.
* Arbitrary recurrences in the graph
