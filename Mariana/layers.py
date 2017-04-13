from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

import theano, numpy, time, uuid
import theano.tensor as tt

import Mariana.abstraction as MABS
import Mariana.activations as MA
import Mariana.initializations as MI
import Mariana.settings as MSET
import Mariana.custom_types as MTYPES

import Mariana.network as MNET
import Mariana.wrappers as MWRAP
import Mariana.candies as MCAN

__all__=["Layer_ABC", "DenseOutput_ABC", "Dense", "Output_ABC", "Input", "Hidden", "Addition", "Embedding", "SoftmaxClassifier", "Regression", "Autoencode"]

class ClassName(object):
    """docstring for ClassName"""
    def __init__(self, arg):
        self.arg = arg
        
class Layer_ABC(MABS.Abstraction_ABC) :
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

    def __init__(self, activation=MA.Pass(), regularizations=[], initializations=[], learningScenari=[], decorators=[], name=None, maxInConnections=1):
        super(Layer_ABC, self).__init__()
        # self.isLayer=True
        self.maxInConnections=maxInConnections

        #a unique tag associated to the layer
        self.appelido=str(uuid.uuid1())

        if name is not None :
            self.name=name
        else :
            self.name="%s_%s" %(self.__class__.__name__, self.appelido)

        self.types=None

        self.inputs=MTYPES.Variable()
        self.outputs=MTYPES.Variable()
        self.outputs_preactivation=MTYPES.Variable()

        self.abstractions={
            "activation": activation,
            "regularizations": regularizations,
            "decorators": decorators,
            "initializations": initializations,
            "learningScenari": learningScenari,
        }
        
        self.network=MNET.Network()
        self.network._addLayer(self)

        self._inputRegistrations=set()

        self._mustInit=True
        self._mustReset=True
        self._decorating=False

    # def getParameters(self) :
    #     """returns the layer's parameters as dictionary"""
    #     return self.parameters

    # def getParameter(self, pname) :
    #     return self.parameters[pname]

    # def getParameters(self) :
    #     """returns the layer's parameters"""
    #     return self.getParameters().values()

    # def getParameterNames(self) :
    #     """returns the layer's parameters names"""
    #     return self.getParameters().keys()

    def getParameterShape_abs(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplementedError("Should be implemented in child: %s" % self.name)

    def clone(self, **kwargs) :
        """Returns a free layer with the same parameters."""
        creationArguments=dict(self.creationArguments["kwargs"])
        creationArguments.update(kwargs)
        newLayer=self.__class__(*self.creationArguments["args"], **creationArguments)
        
        for k, v in self.getParameters().iteritems() :
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
    
    def getInLayers(self) :
        """get the list of all incoming layers"""
        return self.network.inConnections[self]

    def getOutLayers(self) :
        """get the list of all outgoing layers"""
        return self.network.outConnections[self]

    def getShape_abs(self) :
        """returns the shape of the layer. The first dimention is for the minibatch"""
        raise NotImplementedError("Should be implemented in child: %s" % self.name)        
    
    def getDimentionality(self) :
        """returns the layer intrinsic dimentionality (without the minibatch dimention), by default len(shape)-1"""
        return len(self.getShape_abs()) -1

    def _initParameters(self, forceReset=False) :
        """creates the parameters if necessary (self._mustRest == True)"""
        if self._mustReset or forceReset :        
            for init in self.abstractions["initializations"] :
                init.apply(self)
        self._mustReset=False

    def setInputs_abs(self) :
        """Sets the inputs to the layer"""
        raise NotImplementedError("Should be implemented in child: %s" % self.name)

    def setOutputs_abs(self) :
        """Defines the outputs and outputs["test"] of the layer before the application of the activation function. This function is called by _init() ans should be written in child."""
        raise NotImplementedError("Should be implemented in child: %s" % self.name)

    def _decorate(self) :
        """applies decorators"""
        for d in self.abstractions["decorators"] :
            d.apply(self)

    def _activate(self) :
        """applies activation"""
        self.outputs_preactivation["train"]=self.outputs["train"]
        self.outputs_preactivation["test"]=self.outputs["test"]

        self.abstractions["activation"].apply(self, self.outputs)

    def _setTheanoFunctions(self) :
        """Creates propagate/propagateTest theano function that returns the layer's outputs.
        propagateTest returns the outputs["test"], some decorators might not be applied.
        This is called after decorating"""
        self.propagate = {}
        self.propagate["train"]=MWRAP.TheanoFunctionHandle("propagate", self, self.outputs, stream="train", allow_input_downcast=True)
        self.propagate["test"]=MWRAP.TheanoFunctionHandle("propagateTest", self, self.outputs, stream="test", allow_input_downcast=True)
        
    def _parametersSanityCheck(self) :
        "perform basic parameter checks on layers, automatically called on initialization"
        for k, v in self.parameters.iteritems() :
            if v is None :
                raise ValueError("Parameter '%s' of layer '%s' has invalid value %s" % (k, self.name, self.getParameterShape(k)))
        
            if None in self.getParameterShape(k) :
                raise ValueError("Parameter '%s' of layer '%s' has invalid shape %s. That can cause initializations to crash." % (k, self.name, self.getParameterShape(k)))
        
    def _outputsSanityCheck(self) :
        "perform basic output checks on layers, automatically called on initialization"
        for s in self.outputs.streams :
            v = self.outputs[s]
            if not v :
                raise AttributeError("Layers %s has invalid ouput: %s for stream: %s" % (self.name, v, s))
 
    def _setTypes(self) :
        """Brows layer parameters to see its a input, hidden or output layer"""
        self.types = set()
        for k, v in self.__dict__.iteritems() :
            if isinstance(v, MTYPES.Inputs) :
                self.types.add(MSET.TYPE_INPUT_LAYER)
            
            if isinstance(v, MTYPES.Targets) :
                self.types.add(MSET.TYPE_OUTPUT_LAYER)

            if len(self.getInLayers()) > 0 and len(self.network.outConnections[self]) :
                self.types.add(MSET.TYPE_HIDDEN_LAYER)

    def _initA(self) :
        """Initialize the essential attributes of the layer such as: outputs and activations. This function is automatically called before train/test etc..."""
        if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.getInLayers()) ) :
            self._whateverFirstInit()
            self.setInputs_abs()
            self._parametersSanityCheck()
            self._initParameters()
            self.setOutputs_abs()
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
        
    def maleConnect(self, layer) :
        """What happens to A when A > B"""
        pass

    def _femaleConnect(self, layer) :
        if self.maxInConnections is not None :
            if len(self.getInLayers()) > self.maxInConnections :
                raise ValueError("Layer %s can have no more than %s incomming connections" % (layer.name, self.maxInConnections))
        return self.femaleConnect(layer)

    def femaleConnect(self, layer) :
        """What happens to B when A > B"""
        pass

    def connect(self, layer) :
        """Connect the layer to another one. Using the '>' operator to connect two layers actually calls this function.
        This function returns the resulting network"""
        self.maleConnect(layer)
        layer._femaleConnect(self)
        self.network.merge(self, layer)

        return self.network

    def _dot_representation(self) :
        "returns the representation of the node in the graph DOT format"
        return '[label="%s: %s"]' % (self.name, self.getShape_abs())

    def __gt__(self, pathOrLayer) :
        """Alias to connect, make it possible to write things such as layer1 > layer2"""
        return self.connect(pathOrLayer)

    def __repr__(self) :
        return "(Mariana %s '%s': %s )" % (self.__class__.__name__, self.name, self.getShape_abs())

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

class Input(Layer_ABC) :
    """"General representation of an input layer for creating taylored input layers on the fly.
    This one is not abstract an can instanciated wthout risk.
    
    :param int/tuple shape: the shape of the layer, can be a int if its just to specify a number of units. Do not add the minibatch to the shape. Mariana is smart enough to add it for you.
    :param dtype: the numpy data type 
    """
    def __init__(self, shape, name=None, dtype=theano.config.floatX,  **kwargs) :
        # super(Input, self).__init__(layerTypes=[MSET.TYPE_INPUT_LAYER], name=name, **kwargs)
        super(Input, self).__init__(name=name, **kwargs)
        
        if isinstance(shape, int) :
            self.shape = (1, shape)
        else :
            self.shape = tuple([1].extend(list(shape)))
        
        self.broadcastable = [s == 1 for s in self.shape]
        # self.inputs = MTYPES.Inputs(tt.TensorType, dtype=theano.config.floatX, name="Inp_%s" % self.name, broadcastable=self.broadcastable)
        self.inputs = MTYPES.Inputs(tt.matrix, name="Inp_%s" % self.name)
    
    def setInputs_abs(self) :
        pass

    def getShape_abs(self) :
        return self.shape

    def setOutputs_abs(self) :
        "initializes the output to be the same as the inputs"
        self.outputs["train"]=self.inputs["train"]
        self.outputs["test"]=self.inputs["test"]

    def femaleConnect(self, *args) :
        raise ValueError("Nothing can be connected to an input layer")

class Embedding(Layer_ABC) :
    """Embeddings are learned representations of the inputs that are much loved in NLP.
    This layer will take care of creating the embeddings and optimizing them. It can either be used as an input layer or as hidden layer"""

    def __init__(self, nbDimentions, dictSize, zeroForNull=False, initializations=[MI.Uniform('embeddings', small=True)], **kwargs) :
        """
        :param int nbDimentions: the number of dimentions in wich to encode each word.
        :param int dictSize: the total number of words.
        :param bool zeroForNull: if True the dictionnary will be augmented by one elements at te begining (index=0) whose parameters will always be vector of zeros. This can be used to selectively mask some words in the input, but keep in mind that the index for the first word is moved to one.
        :param int size: the size of the input vector (if your input is a sentence this should be the number of words in it). You don't have to provide a value in the embedding layer is a hidden layer
        
        """

        # super(Embedding, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], initializations=initializations, **kwargs)
        super(Embedding, self).__init__(initializations=initializations, **kwargs)

        self.zeroForNull=zeroForNull

        self.setHP("dictSize", dictSize)
        self.setHP("nbDimentions", nbDimentions)

        self.setP("embeddings", MTYPES.Parameter(name="%s.%s" % (self.name, "embeddings")))

        self.inputs=MTYPES.Variable()

    def femaleConnect(self, layer) :
        if layer.getDimentionality() != 1 :
            raise ValueError("Input layer must be a vector, got %s dimentions" % layer.getDimentionality())
        self.nbInputs = layer.getShape_abs()[1]

    def getShape_abs(self) :
        return (1, self.nbInputs * self.nbDimentions)

    def setInputs_abs(self) :
        inpLayer = self.getInLayers()[0]
        if inpLayer.outputs["train"].dtype.find("int") != 0 :
            self.inputs["train"] = tt.cast(inpLayer.outputs["train"], dtype=MSET.INTX)
            self.inputs["test"] = tt.cast(inpLayer.outputs["test"], dtype=MSET.INTX)
        else :
            self.inputs["train"] = inpLayer.outputs["train"]
            self.inputs["test"] = inpLayer.outputs["test"]
        
    def setOutputs_abs(self) :
        if self.zeroForNull :
            self.null=numpy.zeros((1, self.nbDimentions))
            embs = self.parameters["embeddings"].getValue()
            self.parameters["embeddings"].setValue( tt.concatenate( [self.null, embs], axis=0) )
       
        self.preOutputs=self.parameters["fullEmbeddings"][self.inputs["train"]]
        self.outputs["train"]=self.preOutputs.reshape((self.inputs["train"].shape[0], self.nbOutputs))
        self.outputs["test"]=self.preOutputs.reshape((self.inputs["test"].shape[0], self.nbOutputs))

# class Addition(Layer_ABC):
#     """Adds up the values of all afferent layers"""

#     def __init__(self, name=None, **kwargs):
#         super(Addition, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], size=None, name=name, **kwargs)

#     def femaleConnect(self, layer) :
#         if self.nbInputs is None :
#             self.nbInputs=layer.nbOutputs
#         elif self.nbInputs != layer.nbOutputs :
#             raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )
#         self.nbOutputs=layer.nbOutputs

#     def setInputs_abs(self) :
#         """set the number of inputs and outputs"""
#         inputs = 0
#         testInputs = 0
#         for l in self.getInLayers() :
#             inputs += l.outputs["train"]
#             testInputs += l.outputs["test"]

#         self.inputs["train"] = inputs
#         self.inputs["test"] = testInputs

#     def setOutputs_abs(self) :
#         self.outputs["train"]=self.inputs["train"]
#         self.outputs["test"]=self.inputs["test"]

class Pass(Layer_ABC) :
    def __init__(self, name=None, **kwargs):
        # super(Pass, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], size=None, name=name, **kwargs)
        super(Pass, self).__init__(size=None, name=name, **kwargs)
        self.shape = None

    def getShape_abs(self) :
        return self.getInLayers().values()[0].getShape_abs()

    def setOutputs_abs(self) :
        layer = self.getInLayers().values()[0]
        for f in layer.outputs.streams :
            self.outputs[f] = layer.outputs[f]

class Dense(Layer_ABC) :
    """A layer with weigth and bias. If would like to disable either one of them do not provide an initialization"""

    def __init__(self, nbUnits, initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)], **kwargs) :
        # super(Dense, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], initializations=initializations, **kwargs)
        super(Dense, self).__init__(initializations=initializations, **kwargs)
        # self.inputs["test"]=None
        self.addParameters({
            "W": MTYPES.Parameter("%s.W" % (self.name)),
            "b": MTYPES.Parameter("%s.b" % (self.name))
        })

        self.setHP("nbUnits", nbUnits)
        self.nbInputs=None

    def femaleConnect(self, layer) :
        if layer.getDimentionality() != 1 :
            raise ValueError("Input layer must be a vector, got %s dimentions" % layer.getDimentionality())
        self.nbInputs = layer.getShape_abs()[1]

    def getShape_abs(self) :
        """defines the number of inputs"""
        return (1, self.nbUnits)        

    def setInputs_abs(self) :
        l = list(self.getInLayers())[0]
        for s in l.outputs.streams :
            self.inputs[s] = l.outputs[s]

    def setOutputs_abs(self) :
        """Defines, self.outputs["train"] and self.outputs["test"]"""
        if self.parameters["W"] is not None:
            for s in self.inputs.streams :
                self.outputs[s]=tt.dot(self.inputs[s], self.parameters["W"]())
            
        if self.parameters["b"] is not None:
            for s in self.inputs.streams :
                self.outputs[s]=self.outputs[s] + self.parameters["b"]()

    def getParameterShape(self, param) :
        if param == "W" :
            return (self.nbInputs, self.nbUnits)
        elif param == "b" :
            return (self.nbUnits,)
        else :
            raise ValueError("Unknown parameter: %s" % param)

    # def getW(self) :
    #     """Return the weight values"""
    #     try :
    #         return self.parameters["W"].get_value()
    #     except AttributeError :
    #         raise ValueError("It looks like the network has not been initialized yet")

    # def getb(self) :
    #     """Return the bias values"""
    #     try :
    #         return self.parameters["b"].get_value()
    #     except AttributeError :
    #         raise ValueError("It looks like the network has not been initialized yet")

Hidden = Dense
    
        
class Output_ABC(Layer_ABC) :
    """The interface that every output layer should expose.
    If backTrckAll is set to True, the output layer will consider all layers of the network as its dependencies and update them when necessary.
    The default behaviour is to only consider the layers that are parts of branches that lead to the output layer.

    This interface also provides the model functions::
        * train: upadates the parameters and returns the cost
        * test: returns the cost, ignores trainOnly decoartors
        """

    def __init__(self, cost, backTrckAll=False, **kwargs) :
        # super(Output_ABC, self).__init__(layerTypes=[MSET.TYPE_OUTPUT_LAYER], **kwargs)
        super(Output_ABC, self).__init__(**kwargs)
        self.targets=MTYPES.Targets()
        self.dependencies=OrderedDict()
        self.backTrckAll=backTrckAll

        self.cost=cost
        self.loss=None
        # self.updates=None
        # self._mustRegularize=True
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

    # def _applyRegularizations(self, force=False) :
    #     """Defines the regularizations to be added to the cost"""
    #     if self._mustRegularize or force :
    #         self._backTrckDependencies()
    #         for l in self.dependencies.itervalues() :
    #             # for sc in self.abstractions["scenari"] :
    #                 # l.pushLearningScenario(sc)
    #             try :
    #                 for reg in l.abstractions["regularizations"] :
    #                     self.loss["train"] += reg.apply(l)
    #                     # reg.apply(l, self.loss)
    #             except AttributeError :
    #                 pass
    #     self._mustRegularize=False

    def _setTheanoFunctions(self) :
        """
        Sets all the theano function.
        Calls self._setLosses() before if either self.cost or self.testCost is None.
        self._applyRegularizations()
        Calls self._setUpdates() if self.updates is None.
        """
        self._backTrckDependencies()
        super(Output_ABC, self)._setTheanoFunctions()
        self.loss = MTYPES.Losses(self, self.cost, self.targets, self.outputs)
        # self._applyRegularizations()
        
        # if self.updates is None :
        #     self._setUpdates()

        self.train = MWRAP.TheanoFunctionHandle("train", self, self.loss, stream="train", update=True, allow_input_downcast=True)
        # self.train = MWRAP.TheanoFunctionHandle("train", self, self.loss["train"], stream="train", updates=self.updates, allow_input_downcast=True)
        self.test = MWRAP.TheanoFunctionHandle("test", self, self.loss, stream="test", allow_input_downcast=True)

class DenseOutput_ABC(Output_ABC, Dense):
    """Generic output layer with weight and bias"""
    def __init__(self, nbUnits, cost, learningScenari, activation, **kwargs):
        super(DenseOutput_ABC, self).__init__(nbUnits=nbUnits, cost=cost, learningScenari=learningScenari, activation=activation, **kwargs)

    def setOutputs_abs(self) :
        Dense.setOutputs_abs(self)
 
class SoftmaxClassifier(DenseOutput_ABC) :
    """A softmax (probabilistic) Classifier"""
    def __init__(self, nbUnits, cost, learningScenari, temperature=1, **kwargs) :
        super(SoftmaxClassifier, self).__init__(nbUnits, cost=cost, learningScenari=learningScenari, activation=MA.Softmax(temperature=temperature), **kwargs)
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

class Regression(DenseOutput_ABC) :
    """For regressions, works great with a mean squared error cost"""
    def __init__(self, nbUnits, activation, learningScenari, cost, name=None, **kwargs) :
        super(Regression, self).__init__(nbUnits, activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targets=tt.matrix(name="targets")

class Autoencode(DenseOutput_ABC) :
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
