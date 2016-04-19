CHANGELOG
=========

1.0.0rc:
--------

* The begining of a new era for Mariana.
* There is as new abstraction type: initalialization (initializations.py).
* Added batch normalization layer.
* New Layer_ABC functions: getParameter, getParameterDict, getParameterNames, getParameterShape. The last one must be definded for initializations to work.
* GlorotTanhInit is now an initialization.
* Most abstractions now have a common interface.
* More consistent and sane layer implementation.
* All layers now have: activation, regularizations, initializations, learningScenario, decorators and name.
* Layer types have been moved to Network.
* Classifier_ABC is no more.
* New abstract class WeightBias_ABC.
* Networks now have a log, that can be pretty printed using printLog().
* saveOutputs argument is no more
* All layers now have propagate() model function that returns their outputs.
* Output layers can now also serve as hidden layers.
* ToHidden() and toOutput() are no more.
* SoftmaxClassifier() now has an accuracy function.
* AutoEncoder layer now takes a layer name as argument.
* Functions to save parameters of a network in npy or HDF5 formats.
* Save() is now based on clone()  and can now handle many layers and still uses pickle (Yeah I said that I am going to do something using HDF5 and JSON, but it is not worth the trouble).
* CloneBare() is no more.
* Clone() can now clone any layer based on the constructor arguments but you need to call the introspective self._setCreationArguments() at the end of the constructor. 
* Network.load() to load models saved by save().
* Embedding for Conv nets.
* Added example for hierarchical softmax.
* Many other things and little adjustements that make the code more beautiful.
