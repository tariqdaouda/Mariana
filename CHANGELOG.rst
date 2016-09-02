CHANGELOG
=========

1.0.3rc:
--------

* Null cost redifined as a function of outputs and targets
* GeometricalEarlyStopping can now work descending (default) or ascending
* Better abstraction of saving criteria
* Minor refactoring of GGPlot2 recorder
* Added SavePeriod to periodically save the model
* Embedding has now a paramater that allows the masking of inputs by using the label 0 
* Scale in softmax
* Mandatory setCreationArgument() is gone for good
* New saving method allows for Layers to be passed as constructor arguments
* clone() now uses deepcopy and is no longer used in saving
 
1.0.2rc:
--------

* Fixed multiple inputs and added test
* Minor doc updates and cleaning
* printLog() of network works even in the model does not compile, and shows the exception message at the end

1.0.1rc:
--------
* Theano functions can now have several outputs. Model function no longer return an array, but an ordered dict where each key conrrespond to a given output
* Theano function wrapper will now need more arguments, such as the names given to each output
* Added accuracy functions such as: testAndAccuracy, and trainAndAccuracy that return both the score and the accuracy
* Updated trainer/recorder/stopCriteria to support function multiple outputs. They now have more parameters
* trainer now lets you define which function to use for train, test and validation 
* Added SavingRules (children of SavingRule_ABC) to decide when the model should be saved by the recorder. SavingRules are passed through the argument whenToSave
* Created SaveMin and SaveMax SavingRules
* EndOfTraining exceptions are now handeled independently from other exceptionin trainer.

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
