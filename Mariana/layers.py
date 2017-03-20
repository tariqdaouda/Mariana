from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

import theano, numpy, time
import theano.tensor as tt

import Mariana.activations as MA
import Mariana.initializations as MI
import Mariana.settings as MSET
import Mariana.custom_types as MTYPES

import Mariana.network as MNET
import Mariana.wrappers as MWRAP
import Mariana.candies as MCAN

__all__=["Layer_ABC", "WeightBiasOutput_ABC", "WeightBias_ABC", "Output_ABC", "Input", "Hidden", "Addition", "Embedding", "SoftmaxClassifier", "Regression", "Autoencode"]

class Layer_ABC(object) :
    "The interface that every layer should expose"

    __metaclass__=ABCMeta

    def __new__(cls, *args, **kwargs) :
        import inspect
        
        obj=super(Layer_ABC, cls).__new__(cls, *args, **kwargs)
        # argNames=inspect.getargspec(obj.__init__)[0][1:] #remove self
        
        finalKwargs={}
        for k, v in kwargs.iteritems() :
            finalKwargs[k]=v

        obj.creationArguments={
            "args": args,
            "kwargs": finalKwargs,
        }

        return obj

    def __init__(self,
        size,
        layerTypes,
        activation=MA.Pass(),
        regularizations=[],
        initializations=[],
        learningScenari=[],
        decorators=[],
        name=None
    ):

        self.isLayer=True

        #a unique tag associated to the layer
        self.appelido=numpy.random.random()

        if name is not None :
            self.name=name
        else :
            self.name="%s_%s" %(self.__class__.__name__, self.appelido)

        self.types=layerTypes

        self.nbInputs=None
        self.inputs=MTYPES.Variable()

        self.nbOutputs=size
        self.outputs=MTYPES.Variable()
        self.outputs_preactivation=MTYPES.Variable()

        self.abstractions={
            "activation": activation,
            "regularizations": regularizations,
            "decorators": decorators,
            "initializations": initializations,
            "scenari": learningScenari,
        }
        self.parameters = {}
        
        self.network=MNET.Network()
        self.network._addLayer(self)

        self._inputRegistrations=set()

        self._mustInit=True
        self._mustReset=True
        self._decorating=False

    def getParameterDict(self) :
        """returns the layer's parameters as dictionary"""
        from theano.compile import SharedVariable
        res={}
        for k, v in self.parameters.iteritems() :
            if isinstance(v, SharedVariable) :
                res[k]=v
        return res

    def getParameters(self) :
        """returns the layer's parameters"""
        return self.getParameterDict().values()

    def getParameterNames(self) :
        """returns the layer's parameters names"""
        return self.getParameterDict().keys()

    def getParameterShape(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplemented("Should be implemented in child")

    def getOutputShape(self):
        """returns the shape of the outputs"""
        return (self.nbOutputs, )

    def clone(self, **kwargs) :
        """Returns a free layer with the same parameters."""
        creationArguments=dict(self.creationArguments["kwargs"])
        creationArguments.update(kwargs)
        newLayer=self.__class__(*self.creationArguments["args"], **creationArguments)
        
        for k, v in self.getParameterDict().iteritems() :
            setattr(newLayer, k, v)
            newLayer._mustReset=False
        return newLayer

    def _registerInput(self, inputLayer) :
        "Registers a layer as an input to self. This function is first called by input layers. Initialization can only start once all input layers have been registered"
        self._inputRegistrations.add(inputLayer.name)

    def _whateverFirstInit(self) :
        """The first function called during initialization. Does nothing by default, you can put in it
        whatever pre-action you want performed on the layer prior to normal initialization"""
        pass

    def _whateverLastInit(self) :
        """The last function called during initialization. Does nothing by default, you can put in it
        whatever post-action you want performed on the layer post normal initialization"""
        pass
    
    def _setShape(self) :
        """last chance to define the layer's shape before parameter initialization"""
        pass

    def _initParameters(self, forceReset=False) :
        """creates the parameters if necessary (self._mustRest == True)"""
        self._setShape()
        if self._mustReset or forceReset :        
            for init in self.abstractions["initializations"] :
                init.apply(self)
        self._mustReset=False

    def initParameter(self, parameter, value) :
        """Initialize a parameter, raise value error if already initialized"""
        # print self, parameter
        if parameter not in self.parameters :
            raise ValueError("Layer '%s' has no parameter '%s'. Add it to self.parameters dict and give it a None value." % (self.name, parameter) )
            
        if self.parameters[parameter] is None:
            self.parameters[parameter] = value
        else :
            raise ValueError("Parameter '%s' of layer '%s' has already been initialized" % (parameter, self.name) )

    def updateParameter(self, parameter, value) :
        """Update the value of an already initialized parameter. Raise value error if the parameter has not been initialized"""
        if parameter not in self.getParameterDict().keys() :
            raise ValueError("Parameter '%s' has not been initialized as parameter of layer '%s'" % (parameter, self.name) )
        else :
            self.parameters[parameter] = value

    #theano hates abstract methods defined with a decorator
    def _setInputs(self) :
        """Sets the inputs to the layer. Default behaviour is concatenation"""
        self.nbInputs=0
        outs=[]
        testOuts=[]
        for l in self.network.inConnections[self] :
            self.nbInputs += l.nbOutputs  
            outs.append(l.outputs["train"])
            testOuts.append(l.outputs["test"])
        
        self.inputs["train"]=tt.concatenate( outs, axis=1 )
        self.inputs["test"]=tt.concatenate( testOuts, axis=1 )

    def _setOutputs(self) :
        """Defines the outputs and outputs["test"] of the layer before the application of the activation function. This function is called by _init() ans should be written in child."""
        raise NotImplemented("Should be implemented in child")

    def _decorate(self) :
        """applies decorators"""
        for d in self.abstractions["decorators"] :
            d.apply(self)

    def _activate(self) :
        """applies activation"""
        self.outputs_preactivation["train"]=self.outputs["train"]
        self.outputs_preactivation["test"]=self.outputs["test"]

        self.outputs["train"]=self.abstractions["activation"].apply(self, self.outputs_preactivation["train"], 'training')
        self.outputs["test"]=self.abstractions["activation"].apply(self, self.outputs_preactivation["test"], 'testing')

    def _setTheanoFunctions(self) :
        """Creates propagate/propagateTest theano function that returns the layer's outputs.
        propagateTest returns the outputs["test"], some decorators might not be applied.
        This is called after decorating"""
        self.propagate=MWRAP.TheanoFunctionHandle("propagate", self, self.outputs["train"], stream="train", allow_input_downcast=True)
        self.propagateTest=MWRAP.TheanoFunctionHandle("propagateTest", self, self.outputs["test"], stream="test", allow_input_downcast=True)
        
    def _parametersSanityCheck(self) :
        "perform basic parameter checks on layers, automatically called on initialization"
        for k, v in self.getParameterDict().iteritems() :
            try :
                if None in self.getParameterShape(k) :
                    raise ValueError("Parameter '%s' of layer '%s' has invalid shape %s. That can cause initializations to crash." % (k, self.name, self.getParameterShape(k)))
            except :
                message="Warning! Unable to get shape of parameter '%s' of layer '%s'. That can cause initializations to crash." % (k, self.name)
                self.network.logLayerEvent(self, message, {})
                if MSET.VERBOSE :
                    print(message)

    def _outputsSanityCheck(self) :
        "perform basic output checks on layers, automatically called on initialization"
        try :
            if self.outputs["test"] is None :
                raise AttributeError("Attribute 'outputs['test']' of layer '%s' has None value. This attribute defines the test output of the layer, usually without regularizations" % self.name)
        except AttributeError :
                raise AttributeError("Attribute 'outputs['test']' of layer '%s' is not defined. This attribute defines the test output of the layer, usually without regularizations" % self.name)

        try :
            if self.outputs["train"] is None :
                raise AttributeError("Attribute 'outputs' of layer '%s' has None value. This attribute defines the train output of the layer, usually with regularizations" % self.name)
        except AttributeError :
                raise AttributeError("Attribute 'outputs' of layer '%s' is not defined. This attribute defines the train output of the layer, usually with regularizations" % self.name)

    def pushLearningScenario(self, sc) :
        """Adds a new top optimizer"""
        self.abstractions["scenari"].insert(0, sc)

    def _initA(self) :
        """Initialize the essential attributes of the layer such as: outputs and activations. This function is automatically called before train/test etc..."""
        if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.network.inConnections[self]) ) :
            self._whateverFirstInit()
            self._setInputs()
            self._parametersSanityCheck()
            self._initParameters()
            self._setOutputs()
            self._decorate()
            self._outputsSanityCheck()
            self._activate()
            
            for l in self.network.outConnections[self] :
                l._registerInput(self)
                l._initA()
            self._mustInit=False

    def _initB(self) :
        """Initialize the fancy attributes of the layer such as: regularizers, decorators and theano functions. This function is automatically called before train/test etc..."""
        # self._listRegularizations()
        self._setTheanoFunctions()
        self._whateverLastInit()
        
    def _maleConnect(self, layer) :
        """What happens to A when A > B"""
        pass

    def _femaleConnect(self, layer) :
        """What happens to B when A > B"""
        pass

    def connect(self, layer) :
        """Connect the layer to another one. Using the '>' operator to connect two layers actually calls this function.
        This function returns the resulting network"""
        self._maleConnect(layer)
        layer._femaleConnect(self)
        self.network.merge(self, layer)

        return self.network

    def _dot_representation(self) :
        "returns the representation of the node in the graph DOT format"
        return '[label="%s: %s"]' % (self.name, self.getOutputShape())

    def __gt__(self, pathOrLayer) :
        """Alias to connect, make it possible to write things such as layer1 > layer2"""
        return self.connect(pathOrLayer)

    def __repr__(self) :
        return "(Mariana %s '%s': %sx%s )" % (self.__class__.__name__, self.name, self.nbInputs, self.nbOutputs)

    def __len__(self) :
        return self.nbOutputs

    def __setattr__(self, k, v) :
        if k == "name" and hasattr(self, k) :
            if len(self.network.layers) > 1 :
                raise ValueError("You can't change the name of a connected layer")
            else :
                object.__setattr__(self, k, v)
                self.network=MNET.Network()
                self.network._addLayer(self)
        
        try :
            deco=self._decorating
        except AttributeError:
            object.__setattr__(self, k, v)
            return

        if deco :
            var=getattr(self, k)
            try :
                var.set_value(numpy.asarray(v, dtype=theano.config.floatX), borrow=True)
                return
            except AttributeError :
                pass

        object.__setattr__(self, k, v)
    
    # def __getattr__(self, k) :
    #     net=object.__getattr__(self, "network")
    #     net.init()
    #     return object.__getattr__(self, k)

class Input(Layer_ABC) :
    "An input layer"
    def __init__(self, size, name=None, **kwargs) :
        super(Input, self).__init__(size, layerTypes=[MSET.TYPE_INPUT_LAYER], name=name, **kwargs)
        self.nbInputs=size
        self.inputs=MTYPES.Inputs(tt.matrix, name="Inp_%s" % self.name)
    
    def _setInputs(self) :
        pass

    def _setOutputs(self) :
        "initializes the output to be the same as the inputs"
        self.outputs["train"]=self.inputs["train"]
        self.outputs["test"]=self.inputs["test"]

    def _femaleConnect(self, *args) :
        raise ValueError("Nothing can be connected to an input layer")

class Embedding(Layer_ABC) :
    """Embeddings are learned representations of the inputs that are much loved in NLP.
    This layer will take care of creating the embeddings and optimizing them. It can either be used as an input layer or as hidden layer"""

    def __init__(self, nbDimentions, dictSize, zeroForNull=False, size=None, initializations=[MI.SmallUniformEmbeddings()], **kwargs) :
        """
        :param int nbDimentions: the number of dimentions in wich to encode each word.
        :param int dictSize: the total number of words.
        :param bool zeroForNull: if True the dictionnary will be augmented by one elements at te begining (index=0) whose parameters will always be vector of zeros. This can be used to selectively mask some words in the input, but keep in mind that the index for the first word is moved to one.
        :param int size: the size of the input vector (if your input is a sentence this should be the number of words in it). You don't have to provide a value in the embedding layer is a hidden layer
        
        """

        super(Embedding, self).__init__(size, layerTypes=[MSET.TYPE_INPUT_LAYER], initializations=initializations, **kwargs)

        self.zeroForNull=zeroForNull

        self.dictSize=dictSize
        self.nbDimentions=nbDimentions

        self.parameters={
            "embeddings":None,
            "fullEmbeddings":None
        }

        self.inputs["train"]=None
        self.inputs["test"]=None
        
        if size is not None :
            self.nbInputs=size
            self.nbOutputs=self.nbDimentions*self.nbInputs    
 
    def _femaleConnect(self, layer) :
        self.types=[MSET.TYPE_HIDDEN_LAYER]
        if not hasattr(self, "nbInputs") or self.nbInputs is None :
            self.nbInputs=layer.nbOutputs
            self.nbOutputs=self.nbDimentions*self.nbInputs
        elif self.nbInputs != layer.nbOutputs :
            raise ValueError("All layers connected to '%s' must have the same number of outputs. Got: %s, previously had: %s" % (self.name, layer.nbOutputs, self.nbInputs) )
    
    def getParameterShape(self, param) :
        if param == "embeddings" :
            return (self.dictSize, self.nbDimentions)
        else :
            raise ValueError("Unknown parameter: %s" % param)

    def _setInputs(self) :
        if len(self.network.inConnections[self]) > 0 :
            super(Embedding, _setInputs)._setInputs()

    def getEmbeddings(self, idxs=None) :
        """returns the embeddings.

        :param list idxs: if provided will return the embeddings only for those indexes
        """
        if not self.parameters["fullEmbeddings"] :
            raise ValueError("It looks like the network has not been initialized yet. Try calling self.network.init() first.")

        try :
            fct=self.parameters["fullEmbeddings"].get_value
        except AttributeError :
            fct=self.parameters["fullEmbeddings"].eval

        if idxs :
            return fct()[idxs]
        return fct()

    def _setOutputs(self) :
        if len(self.network.inConnections[self]) == 0 :
            if self.inputs["train"] is None :
                self.inputs["train"]=tt.imatrix(name="embInp_" + self.name)
                self.inputs["test"]=self.inputs["train"]
        else :
            for layer in self.network.inConnections[self] :
                if layer.outputs["train"].dtype.find("int") != 0 :
                    outs=tt.cast(layer.outputs["train"], dtype=MSET.INTX)
                    testOuts=tt.cast(layer.outputs["test"], dtype=MSET.INTX)
                else :
                    outs=layer.outputs["train"]
                    testOuts=layer.outputs["test"]

                if self.inputs["train"] is None :   
                    self.inputs["train"]=outs
                    self.inputs["test"]=testOuts
                else :
                    self.inputs["train"]+=outs
                    self.inputs["test"]+=testOuts

        if self.zeroForNull :
            self.null=numpy.zeros((1, self.nbDimentions))
            self.parameters["fullEmbeddings"]=tt.concatenate( [self.null, self.parameters["embeddings"]], axis=0)
        else :
            self.parameters["fullEmbeddings"]=self.parameters["embeddings"]
            del(self.parameters["embeddings"])

        self.preOutputs=self.parameters["fullEmbeddings"][self.inputs["train"]]
        self.outputs["train"]=self.preOutputs.reshape((self.inputs["train"].shape[0], self.nbOutputs))
        self.outputs["test"]=self.preOutputs.reshape((self.inputs["test"].shape[0], self.nbOutputs))

class Addition(Layer_ABC):
    """Adds up the values of all afferent layers"""

    def __init__(self, name=None, **kwargs):
        super(Addition, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], size=None, name=name, **kwargs)

    def _femaleConnect(self, layer) :
        if self.nbInputs is None :
            self.nbInputs=layer.nbOutputs
        elif self.nbInputs != layer.nbOutputs :
            raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )
        self.nbOutputs=layer.nbOutputs

    def _setInputs(self) :
        """set the number of inputs and outputs"""
        inputs = 0
        testInputs = 0
        for l in self.network.inConnections[self] :
            inputs += l.outputs["train"]
            testInputs += l.outputs["test"]

        self.inputs["train"] = inputs
        self.inputs["test"] = testInputs

    def _setOutputs(self) :
        self.outputs["train"]=self.inputs["train"]
        self.outputs["test"]=self.inputs["test"]

class Pass(Layer_ABC) :
    def __init__(self, name=None, **kwargs):
        super(Pass, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], size=None, name=name, **kwargs)

    def _femaleConnect(self, layer) :
        if self.nbInputs is None :
            self.nbInputs=layer.nbOutputs
        elif self.nbInputs != layer.nbOutputs :
            raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )
        self.nbOutputs=layer.nbOutputs

    def _setOutputs(self) :
        for layer in self.network.inConnections[self] :
            if self.inputs["train"] is None :
                self.inputs["train"]=layer.outputs["train"]
            else :
                self.inputs["train"] += layer.outputs["train"]

        self.outputs["train"]=self.inputs["train"]
        self.outputs["test"]=self.inputs["train"]

class WeightBias_ABC(Layer_ABC) :
    """A layer with weigth and bias. If would like to disable either one of them do not provide an initialization"""

    def __init__(self, size, layerTypes, initializations=[MI.SmallUniformWeights(), MI.ZeroBias()], **kwargs) :
        super(WeightBias_ABC, self).__init__(size, layerTypes=layerTypes, initializations=initializations, **kwargs)
        self.inputs["test"]=None
        self.parameters={
            "W": None,
            "b": None
        }

    def _setShape(self) :
        """defines the number of inputs"""
        self.nbInputs=None
        for layer in self.network.inConnections[self] :
            if self.nbInputs is None :
                self.nbInputs=layer.nbOutputs
            else :
                self.nbInputs += layer.nbOutputs
                # raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )

    def _setOutputs(self) :
        """Defines, self.outputs["train"] and self.outputs["test"]"""

        self.outputs["train"]=self.inputs["train"]
        self.outputs["test"]=self.inputs["test"]
        
        if self.parameters["W"] is not None:
            self.outputs["train"]=tt.dot(self.inputs["train"], self.parameters["W"])
            self.outputs["test"]=tt.dot(self.inputs["test"], self.parameters["W"])
            
        if self.parameters["b"] is not None:
            self.outputs["train"]=self.outputs["train"] + self.parameters["b"]
            self.outputs["test"]=self.outputs["test"] + self.parameters["b"]

    def getParameterShape(self, param) :
        if param == "W" :
            return (self.nbInputs, self.nbOutputs)
        elif param == "b" :
            return (self.nbOutputs,)
        else :
            raise ValueError("Unknown parameter: %s" % param)

    def getW(self) :
        """Return the weight values"""
        try :
            return self.parameters["W"].get_value()
        except AttributeError :
            raise ValueError("It looks like the network has not been initialized yet")

    def getb(self) :
        """Return the bias values"""
        try :
            return self.parameters["b"].get_value()
        except AttributeError :
            raise ValueError("It looks like the network has not been initialized yet")

class Hidden(WeightBias_ABC) :
    "A hidden layer with weigth and bias"
    def __init__(self, size, **kwargs) :
        super(Hidden, self).__init__(size, layerTypes=[MSET.TYPE_HIDDEN_LAYER], **kwargs)

class Output_ABC(Layer_ABC) :
    """The interface that every output layer should expose.
    If backTrckAll is set to True, the output layer will consider all layers of the network as its dependencies and update them when necessary.
    The default behaviour is to only consider the layers that are parts of branches that lead to the output layer.

    This interface also provides the model functions::
        * train: upadates the parameters and returns the cost
        * test: returns the cost, ignores trainOnly decoartors
        """

    def __init__(self, size, cost, backTrckAll=False, **kwargs) :
        super(Output_ABC, self).__init__(size, layerTypes=[MSET.TYPE_OUTPUT_LAYER], **kwargs)
        self.targets=MTYPES.Targets()
        self.dependencies=OrderedDict()
        self.backTrckAll=backTrckAll

        self.cost=cost
        self.loss=None
        self.updates=None
        self._mustRegularize=True
        self.dependencies=None

    def _backTrckDependencies(self, force=False) :
        """Finds all the hidden layers the ouput layer is influenced by"""
        def _bckTrk(deps, layer) :
            for l in self.network.inConnections[layer] :
                deps[l.name]=l
                _bckTrk(deps, l)
            return deps

        if self.dependencies is None or force :
            self.dependencies={}
            if self.backTrckAll == True:
                self.dependencies=dict(self.network.layers)
                del self.dependencies[self.name]
            else:
                self.dependencies=_bckTrk(self.dependencies, self)

    def _applyRegularizations(self, force=False) :
        """Defines the regularizations to be added to the cost"""
        if self._mustRegularize or force :
            self._backTrckDependencies()
            for l in self.dependencies.itervalues() :
                for sc in self.abstractions["scenari"] :
                    l.pushLearningScenario(sc)
                try :
                    for reg in l.abstractions["regularizations"] :
                        self.loss["train"] += reg.apply(l)
                except AttributeError :
                    pass
        self._mustRegularize=False

    def _setUpdates(self) :
        """Defines parameter updates according to training scenari"""
        self._backTrckDependencies()
        self.updates = MWRAP.Updates(self, self.loss["train"])

    def _setTheanoFunctions(self) :
        """
        Sets all the theano function.
        Calls self._setLosses() before if either self.cost or self.testCost is None.
        self._applyRegularizations()
        Calls self._setUpdates() if self.updates is None.
        """
        super(Output_ABC, self)._setTheanoFunctions()
        self.loss = MTYPES.Losses(self, self.cost, self.targets, self.outputs)
        self._applyRegularizations()
        
        if self.updates is None :
            self._setUpdates()

        self.train = MWRAP.TheanoFunctionHandle("train", self, self.loss["train"], stream="train", updates=self.updates, allow_input_downcast=True)
        self.test = MWRAP.TheanoFunctionHandle("test", self, self.loss["test"], stream="test", allow_input_downcast=True)

class WeightBiasOutput_ABC(Output_ABC, WeightBias_ABC):
    """Generic output layer with weight and bias"""
    def __init__(self, size, cost, learningScenari, activation, **kwargs):
        super(WeightBiasOutput_ABC, self).__init__(size=size, cost=cost, learningScenari=learningScenari, activation=activation, **kwargs)

    def _setOutputs(self) :
        WeightBias_ABC._setOutputs(self)
 
class SoftmaxClassifier(WeightBiasOutput_ABC) :
    """A softmax (probabilistic) Classifier"""
    def __init__(self, size, cost, learningScenari, temperature=1, **kwargs) :
        super(SoftmaxClassifier, self).__init__(size, cost=cost, learningScenari=learningScenari, activation=MA.Softmax(temperature=temperature), **kwargs)
        self.targets=MTYPES.Targets(tt.ivector) #tt.ivector(name="targets_" + self.name)

    def _setCustomTheanoFunctions(self) :
        """defines::

            * classify: return the argmax of the outputs applying all the decorators.
            * predict: return the argmax of the test outputs (some decorators may not be applied).
            * classificationAccuracy: returns the accuracy (between [0, 1]) of the model, computed on outputs.
            * predictionAccuracy: returns the accuracy (between [0, 1]) of the model, computed on test outputs.
        """
        Output_ABC._setCustomTheanoFunctions(self)
        clas=tt.argmax(self.outputs["train"], axis=1)
        pred=tt.argmax(self.outputs["test"], axis=1)

        self.classify=MWRAP.TheanoFunction("classify", self, [ ("class", clas) ], flow="train", allow_input_downcast=True)
        self.predict=MWRAP.TheanoFunction("predict", self, [ ("class", pred) ], flow="test", allow_input_downcast=True)

        clasAcc=tt.mean( tt.eq(self.targets, clas ) )
        predAcc=tt.mean( tt.eq(self.targets, pred ) )

        self.classificationAccuracy=MWRAP.TheanoFunction("classificationAccuracy", self, [("accuracy", clasAcc)], additional_input_expressions={ "targets" : self.targets }, flow="train", allow_input_downcast=True)
        self.predictionAccuracy=MWRAP.TheanoFunction("predictionAccuracy", self, [("accuracy", predAcc)], additional_input_expressions={ "targets" : self.targets }, flow="test", allow_input_downcast=True)

        self.trainAndAccuracy=MWRAP.TheanoFunction("trainAndAccuracy", self, [("score", self.cost), ("accuracy", clasAcc)], additional_input_expressions={ "targets" : self.targets },  updates=self.updates, flow="train", allow_input_downcast=True)
        self.testAndAccuracy=MWRAP.TheanoFunction("testAndAccuracy", self, [("score", self.testCost), ("accuracy", predAcc)], additional_input_expressions={ "targets" : self.targets }, flow="test", allow_input_downcast=True)

class Regression(WeightBiasOutput_ABC) :
    """For regressions, works great with a mean squared error cost"""
    def __init__(self, size, activation, learningScenari, cost, name=None, **kwargs) :
        super(Regression, self).__init__(size, activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targets=tt.matrix(name="targets")

class Autoencode(WeightBiasOutput_ABC) :
    """An auto encoding layer. This one takes another layer as inputs and tries to reconstruct its activations.
    You could achieve the same result with a Regression layer, but this one has the advantage of not needing to be fed specific inputs"""

    def __init__(self, targetLayerName, activation, learningScenari, cost, name=None, **kwargs) :
        super(Autoencode, self).__init__(None, activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targetLayerName=targetLayerName

    def _setNbOutputs(self) :
        self.nbOutputs=self.network[self.targetLayerName].nbOutputs
        
    def _initParameters(self, forceReset=False) :
        self._setNbOutputs()
        super(Autoencode, self)._initParameters(forceReset)

    def _whateverFirstInit(self) :
        self.targets=self.network[self.targetLayerName].outputs["train"]
    
    def _setCustomTheanoFunctions(self) :
        super(Autoencode, self)._setCustomTheanoFunctions()
        self.train=MWRAP.TheanoFunction("train", self, [("score", self.cost)], {}, updates=self.updates, flow="train", allow_input_downcast=True)
        self.test=MWRAP.TheanoFunction("test", self, [("score", self.testCost)], {}, flow="test", allow_input_downcast=True)
