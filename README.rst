MARIANA: The Cutest Deep Learning Framework
=============================================
.. image:: https://img.shields.io/badge/python-2.7-blue.svg 
    
**TEMPORARY README FOR MARIANA V2, WIP (BUT ALMOST DONE).**

Mariana is an **efficient language** through which complex deep neural networks can be easily expressed and easily manipulated. It's simple enough for beginners and doesn't get much complicated for seniors.

Intuitive, user-friendly and yet flexible enough for research. It's here to empower **researchers**, **teachers** and **students** alike, while greatly facilitating **AI knowledge transfer** into other domains.

V2's most exciting stuff
=========================

V2 is almost a complete rewite of V1. It is much better.

What's done
-----------

* **Convolutions, Deconvolutions (Transpose convolution)**

* **Lasagne compatible**: Every lasagne layer can be seemlessly imported and used into Mariana
* **Just in time function compilation**: At the heart of Mariana are Theano function. The previous version compiled every function at model initialization. This caused compilation times to be longer than needed. With this version functions are compiled only if needed.
* **Function mixins (My favorite)**: Mariana functions can now be added together! The result is a function that performs the actions of all its components at once. Let's say we have two output layers and want a function that optimises on losses for both outputs. Creating it is as simple as: f = out1.train + out2.train, and then calling f. Mariana will derive f from both functions adding the costs, calculating gradients and updates seemlessly in the background
* **Easy access to gradients and updates**: Just call .getGradients(), .getUpdates() on any function to get a view of either gradients of updates for all parameters.
* **Streams!**: You probably heard of batchnorm... and how it has a different behaviour in training and in testing. Well that simple fact can be the cause of some very messy DL code. With streams all this is over. Streams are parallel universes of execution for functions. You can define your own streams and have as many as you want. For batchnorm it mean that depending on the stream you call your function in (test or train), the behaviour will be different, even though you only changed one word.
* **Chainable optimization rules**: As in the previous version, layers inherit their learning scenari from outputs, but have the possibility to redifine them. This is still true, but rules can now be chained. Here's how to define a layer with fixed bias: l = Dense( learningScenari=[GradientDescent(lr = 0.1), Fixed('b')]) 
* **Much easier to extend**: The (almost) complete rewrite made for a much more cleaner code that is much more easy to extend. It is now much simpler to create your own layers, decorators, etc... Function that you need to implement end with *_abs* and Mariana has whole new bunch of custom type that support streams.
* **New merge layer**: Need a layer that is a linear combination of other layers? The new MergeLayer is perfect for that newLayer = M(layer1 + (layers3 * layer3) + 4 )
* **New concatenation layer**: newLayer = C([Layer1, layer2])
* **Unlimited number of inputs per layer**: Each layer used to be limited to one. Now it is infinit
* **Abstractions are now divided into trainable (layers, decorators, activations) and untrainable (scenari, costs, initializations)**: All trainable abstractions can hold parameters and have untrainable abstractions applied to them. PReLU will finally join ReLU as an activation!


What's almost done
-------------------

* Inclusion of popular recurrences (LSTM, recurent layers)
* New built-in visualisaton: The previous visualization only showed the architecture and layer shapes. The new one will be interactive. It will contain information on all parameters, hyper-parameters as well as user defined notes on the network, layers, or any other abstraction in the network. A great tool for collaboration.

What's next
-----------

* Complete refactorisation of training encapsulation. Training encapsulation was the least popular aspect of Mariana so far. I will completely rewrite it to give it the same level of intuitiveness as the rest of the framework. The next iterration will be a huge improvement.
* Arbitrary recurrences in the graph
