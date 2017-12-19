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

__all__=["Layer_ABC", "ArithmeticMerge", "Input", "Embedding", "Dense", "Hidden",  "Merge", "M", "Concatenation", "C", "DenseOutput_ABC", "Output_ABC", "SoftmaxClassifier", "Regression", "Autoencode"]


class ArithmeticMerge(object):
    """Simple mathematical operations (+, -, *, /) between layers and other stuff"""
    def __init__(self, a, b, op):
        super(ArithmeticMerge, self).__init__()
        
        allowedCls = [ArithmeticMerge, Layer_ABC]
        variables = [a, b]
        preStreams = {}
        preShapes = {}

        self.a = a
        self.b = b
        self.op = op

        count = False
        for var in variables :
            for cl in allowedCls :
                if isinstance(var, cl) :
                    preStreams[var] = set(var.streams)
                    preShapes[var] = tuple(var.getShape_abs())
            
        if len(preStreams) == 2 :
            streams  = preStreams[a] & preStreams[b]
            if len(streams) < 1 :
                raise ValueError("Parameters have no common streams")
            if len(preStreams[a]) != len(streams) :
                print MCAN.warning("Parameter have different streams, will take the intersection")
            self.streams = list(streams)

            if preShapes[a] != preShapes[b] :
                raise ValueError("All parameters nust have the same shape %s != %s" % (preShapes[a], preShapes[b]))        
        elif len(preStreams) == 1 :
            self.streams = list(preStreams.values()[0])
        else :
            raise ValueError("At least a or b must of class: %s" % allowedCls)

        self.shape = preShapes.values()[0]

    def getShape_abs(self):
        return self.shape

    def getDependencies(self) :
        """returns layers needed by the computation"""
        deps = []
        if isinstance(self.a, ArithmeticMerge):
            deps.extend(self.a.getDependencies())
        elif isinstance(self.a, Layer_ABC):
            deps.append(self.a)
            
        if isinstance(self.b, ArithmeticMerge):
            deps.extend(self.b.getDependencies())
        elif isinstance(self.b, Layer_ABC):
            deps.append(self.b)

        return deps

    def getOutputs(self) :
        aOuts = {}
        bOuts = {}
        try:
            aOuts = self.a.getOutputs()
        except Exception as e:
            for s in self.streams :
                aOuts[s] = self.a

        try:
            bOuts = self.b.getOutputs()
        except Exception as e:
            for s in self.streams :
                bOuts[s] = self.b

        outputs = MTYPES.Variable(streams = self.streams)
        for s in self.streams :
            outputs[s] = 0
            if self.op == "+" :
                outputs[s] = aOuts[s] + bOuts[s]
            if self.op == "-" :
                outputs[s] = aOuts[s] - bOuts[s]
            if self.op == "*" :
                outputs[s] = aOuts[s] * bOuts[s]
            if self.op == "/" :
                outputs[s] = aOuts[s] / bOuts[s]
        
        return outputs

    def __add__(self, thing) :
        return ArithmeticMerge(self, thing, "+")

    def __sub__(self, thing) :
        return ArithmeticMerge(self, thing, "-")

    def __mul__(self, thing) :
        return ArithmeticMerge(self, thing, "*")

    def __div__(self, thing) :
        return ArithmeticMerge(self, thing, "/")

    def __repr__(self):
        return "AM(" + repr(self.a) + self.op + repr(self.b) + ")"

class Layer_ABC(MABS.TrainableAbstraction_ABC) :
    "The interface that every layer should expose"

    # __metaclass__=ABCMeta

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

    def __init__(self, activation=MA.Pass(), regularizations=[], initializations=[], learningScenari=[], decorators=[], name=None, maxInConnections=1, streams=["train", "test"]):
        super(Layer_ABC, self).__init__(initializations=[], streams = streams)
        self.maxInConnections=maxInConnections

        #a unique tag associated to the layer
        self.appelido=str(uuid.uuid1())

        if name is not None :
            self.name=name
        else :
            self.name="%s_%s" %(self.__class__.__name__, self.appelido)

        self.inputs=MTYPES.Variable(streams = self.streams)
        self.outputs=MTYPES.Variable(streams = self.streams)
        self.outputs_preactivation=None

        self.abstractions={
            "activation": [activation],
            "regularizations": regularizations,
            "decorators": decorators,
            "initializations": initializations,
            "learningScenari": learningScenari,
        }
        
        self._inputRegistrations=set()
        self._resetNetwork()

    def _resetNetwork(self, fullReset=True, newNetwork = None) :
        if fullReset :
            self._initStatus=0

        if newNetwork is None :
            self.network=MNET.Network()
            self.network._addLayer(self)
        else :
            self.network = newNetwork

    def getLog(self) :
        log = {
            "layer": self.log,
            "abstractions": {}
        }

        for absType, abstractions in self.abstractions.iteritems() :
            log["abstractions"][absType] = []
            for abstraction in abstractions :
                log["abstractions"][absType].append(abstraction.getLog())

        return log

    # def copy(self) :
    #     "return a copy of the layer"
    #     import copy
    #     copy.copy(self)

    def clone(self, **kwargs) :
        """Returns a free layer with the same parameters."""
        import copy

        creationArguments=dict(self.creationArguments["kwargs"])
        creationArguments.update(kwargs)
        newLayer=self.__class__(*self.creationArguments["args"], **creationArguments)
        
        for k, v in self.getParameters().iteritems() :
            newLayer.setP(k, copy.copy(v))
        
        # self._initStatus = 1
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
        """returns the shape of the layer. The first dimension is for the minibatch"""
        raise NotImplementedError("Must be implemented in child: %s" % self.name)        
    
    def getIntrinsicShape(self) :
        """return the shape without the minibatch"""
        return self.getShape_abs()[1:]

    def getDimensionality(self) :
        """returns the layer intrinsic dimensionality (without the minibatch dimension), by default len(shape)-1"""
        return len(self.getShape_abs()) -1

    # def setParameter(self, param, value) :
    #     """Brutally set the value of a parameter. No checks applied"""
    #     if isinstance(value, MTYPES.Parameter) :
    #         self.parameters[param] = value
    #     else :
    #         self.parameters[param].setValue(value)

    def getInputShape(self, layer) :
        selfNdim = len(self.getShape_abs())
        inpNdim = len(layer.getShape_abs())
        
        if selfNdim < inpNdim :
            inShape = layer.getShape_abs()
            flatSize = 1
            for i in xrange(selfNdim-1, inpNdim):
                if inShape[i] is not None :
                    flatSize *= inShape[i]
            newShape = list(self.getShape_abs())
            newShape[0] = -1
            newShape[-1] = flatSize
        elif selfNdim > inpNdim :
            newShape = range(selfNdim)
            for i in xrange(selfNdim - inpNdim) :
                newShape.insert(1, -1)
        else :
            newShape = self.getShape_abs()
        
        return newShape
    
    def setShape_abs(self) :
        """sets the layer shape"""
        pass

    def setInputs(self) :
        """Sets the inputs to the layer and performs of reshaping of the inputs. If there's more that one connection, this function has to redefined"""
        def getInput(layer, stream) :
            if not layer.outputs[stream] :
                raise ValueError("Can't set Inputs for layer '%s', input layer '%s' has '%s' output for stream '%s'." % (self.name, layer.name, layer.outputs[stream], stream))
            
            shape = self.getInputShape(layer)
            if shape != self.getShape_abs() :
                return layer.outputs[stream].reshape(shape)
            else :
                return layer.outputs[stream]

        selfNdim = len(self.getShape_abs())
        layers = list(self.getInLayers())
        if len(layers) > 1 :
            raise AttributeError("There's more than one input layer to %s. You should redefine the setInputs() function" % self)
        
        if len(layers) == 1 :
            layer = list(self.getInLayers())[0]
            for s in self.streams :
                self.inputs[s] = getInput(layer, s)

    def setOutputs_abs(self) :
        """Defines the outputs and outputs["test"] of the layer before the application of the activation function. This function is called by _init() ans should be written in child."""
        raise NotImplementedError("Must be implemented in child: %s" % self.name)

    def _decorate(self) :
        """applies decorators"""
        for s in self.streams :
            for d in self.abstractions["decorators"] :
                d._apply(self, stream=s)

    def _activate(self) :
        """applies activation"""
        self.outputs_preactivation=MTYPES.Variable(streams = self.streams)
        for f in self.streams :
            self.outputs_preactivation[f]=self.outputs[f]
        
        self.abstractions["activation"][0]._apply(self, x=self.outputs)

    def _setTheanoFunctions(self) :
        """Creates propagate theano function that returns the layer's outputs."""
        self.propagate = MWRAP.TheanoFunctionGroup("propagate", self, self.outputs, allow_input_downcast=True)

    def _parametersSanityCheck(self) :
        "perform basic parameter checks on layers, automatically called on initialization"
        super(Layer_ABC, self)._parametersSanityCheck()
        for ab in self.getTrainableAbstractions() :
            ab._parametersSanityCheck()

    def _outputsSanityCheck(self) :
        "perform basic output checks on layers, automatically called on initialization"
        for s in self.outputs.streams :
            v = self.outputs[s]
            if not v and v != 0:
                raise AttributeError("Layers %s has invalid output: %s for stream: %s" % (self.name, v, s))
 
    def getOutputs(self) :
        """return layer outputs"""
        return self.outputs

    def getTypes(self) :
        """Browses layer parameters to see if it is an input, hidden or output layer"""
        types = set()
        for k, v in self.__dict__.iteritems() :
            if isinstance(v, MTYPES.Inputs) :
                types.add(MSET.TYPE_INPUT_LAYER)
            
            if isinstance(v, MTYPES.Targets) :
                types.add(MSET.TYPE_OUTPUT_LAYER)

            if len(self.getInLayers()) > 0 and len(self.network.outConnections[self]) :
                types.add(MSET.TYPE_HIDDEN_LAYER)
        
        return types

    def _initParameters(self, forceReset=False) :
        # if self._initStatus == 0 or self._mustInit or forceReset :
        super(Layer_ABC, self)._initParameters(forceReset=forceReset)
        for absType, abstractions in self.abstractions.iteritems() :
            for abstraction in abstractions :
                if abstraction.isTrainable() :
                    abstraction._initParameters(forceReset=forceReset)
        # self._initStatus = 1

    def initParameters(self, force=False) :
        """Initialize the essential attributes of the layer such as: outputs and activations. This function is automatically called before train/test etc..."""
        if force :
            self._initStatus = 0

        if ( self._initStatus == 0) or self._mustInit :
            self.logEvent("%s: initParameters" % (self.name))
            self.logEvent("%s: _whateverFirstInit" % (self.name))
            self._whateverFirstInit()
            self.logEvent("%s: setInputs" % (self.name))
            self.setInputs()
            self.logEvent("%s: _initParameters" % (self.name))
            self._initParameters()
            self.logEvent("%s: _parametersSanityCheck" % (self.name))
            self._parametersSanityCheck()
            self._initStatus = 1
            
    def _initA(self, force=False, asReasinedInput=False) :
        """Initialize the essential attributes of the layer such as: outputs and activations. This function is automatically called before train/test etc..."""
        if force :
            self._initStatus = 0

        if ( self._initStatus < 2) and ( len(self._inputRegistrations) == len(self.getInLayers()) ) :
            self.logEvent("%s: initA" % (self.name))
            self.initParameters()
            if not self.getShape_abs() :
                self.logEvent("%s: setShape_abs" % (self.name))
                self.setShape_abs()
            self.logEvent("%s: setOutputs_abs" % (self.name))
    
            if not asReasinedInput :
                self.setOutputs_abs()
                for k in self.streams :
                    if self.outputs[k].dtype != theano.config.floatX :
                        self.logEvent("output for stream %s is of the work type (%s vs %s) forcefully casting" % (k, self.outputs[k].dtype, theano.config.floatX))
                        self.outputs[k] = tt.cast(self.outputs[k], theano.config.floatX)
            else :
                shape = self.getShape_abs()
                if len(shape) == 2:
                    typ = tt.matrix
                elif len(shape) == 3:
                    typ = tt.tensor3
                elif len(shape) == 4:
                    typ = tt.tensor4
                elif len(shape) == 5:
                    typ = tt.tensor5
                else :
                    raise ValueError("The maximum shape size allowed is 5")

                self.values = MTYPES.Inputs(typ, streams=self.streams)
                for f in self.streams :
                    self.outputs[f] = self.values[f]

            self.logEvent("%s: _activate" % (self.name))
            self._activate()
            self.logEvent("%s: _decorate" % (self.name))
            self._decorate()
            self.logEvent("%s: _outputsSanityCheck" % (self.name))
            self._outputsSanityCheck()
            self._initStatus = 2

            for l in self.getOutLayers() :
                self.logEvent("register %s as input for %s" % (self.name, l.name))
                l._registerInput(self)
                l._initA(force)

    def _initB(self, force) :
        """Initialize theano functions. This function is automatically called before train/test etc..."""
        # print "\t", self, self._initStatus
        # print self, self._initStatus, len(self._inputRegistrations), len(self.getInLayers())
        if force :
            self._initStatus = 2

        if (self._initStatus == 2) and ( len(self._inputRegistrations) == len(self.getInLayers()) ) :
            self.logEvent("%s: initB" % (self.name))
            self.logEvent("%s: _setTheanoFunctions" % (self.name))
            self._setTheanoFunctions()
            self.logEvent("%s: _whateverLastInit" % (self.name))
            self._whateverLastInit()
            self._initStatus = 3
    
    def isInit(self) :
        return self._initStatus == 3
    
    def toInnput(self) :
        """return an input layer with the same shape as self"""
        inp = Input(self.getShape_abs(), self.name)
        return inp

    def maleConnect(self, layer) :
        """What happens to A when A > B"""
        pass

    def _femaleConnect(self, layer) :
        if self.maxInConnections is not None :
            if len(self.getInLayers()) > self.maxInConnections :
                raise ValueError("Layer %s can have no more than %s incoming connections" % (layer.name, self.maxInConnections))
        return self.femaleConnect(layer)

    def femaleConnect(self, layer) :
        """What happens to B when A > B"""
        pass

    def connect(self, layer) :
        """Connect the layer to another one. Using the '>' operator to connect two layers actually calls this function.
        This function returns the resulting network"""
        self.logEvent("%s > %s" % (self.name, layer.name))
        if layer not in self.getInLayers() :
            self.network.merge(self, layer)
            self.maleConnect(layer)
            layer._femaleConnect(self)

        return self.network

    def toDictionary(self) :
        res = super(Layer_ABC, self).toDictionary()
        try :
            res["shape"] = self.getShape_abs()
        except Exception as e:
            res["shape"] = e.message

        return res

    def getFullParameters(self) :
        from collections import Counter

        params = dict(self.parameters)
        for absType, abstractions in self.abstractions.iteritems() :
            cnt = Counter()
            for abstraction in abstractions :
                cnt[abstraction.name] += 1
                if cnt[abstraction.name] > 1 :
                    nameId = "%d" % (cnt[abstraction.name]-1)
                else :
                    nameId = ""

                for paramName, param in abstraction.getParameters() :
                    k = "{type}.{name}{nameId}.{paramName}".format(type=absType, name=abstraction.name, nameId=nameId, paramName=paramName)
                    params[k] = param

        return params

    def getTrainableAbstractions(self, includeEmpty = False) :
        """return a list of all trainable abstractions within. By default will exclude all abstractions of trainable type
        with no parameters. To change that behaviour set use: includeEmpty=True"""
        res = []
        for k, v in self.abstractions.iteritems() :
            for vv in v :
                if vv.isTrainable() :
                    if len(vv.getParameters()) > 0 or includeEmpty :
                        res.append(vv)
        return res

    def _dot_representation(self) :
        "returns the representation of the node in the graph DOT format"
        return '[label="%s: %s"]' % (self.name, self.getShape_abs())

    def __gt__(self, layer) :
        """Alias to connect, make it possible to write things such as layer1 > layer2"""
        return self.connect(layer)

    def __add__(self, layer) :
        return ArithmeticMerge(self, layer, "+")

    def __sub__(self, layer) :
        return ArithmeticMerge(self, layer, "-")

    def __mul__(self, layer) :
        return ArithmeticMerge(self, layer, "*")

    def __div__(self, layer) :
        return ArithmeticMerge(self, layer, "/")

    def __repr__(self) :
        return "(Mariana %s '%s': %s )" % (self.__class__.__name__, self.name, self.getShape_abs())

    # def __len__(self) :
    #     return self.nbOutputs

    def __setattr__(self, k, v) :
        if k == 'name' and hasattr(self, k) and self.name != v and name is not None :
            raise ValueError("You can't change the name of a layer")    
        MABS.Abstraction_ABC.__setattr__(self, k, v)

class Input(Layer_ABC) :
    """"General representation of an input layer for creating taylored input layers on the fly.
    This one is not abstract an can instanciated wthout risk.
    
    :param int/tuple shape: the shape of the layer, can be a int if its just to specify a number of units. Do not add the minibatch to the shape. Mariana is smart enough to add it for you.
    """
    def __init__(self, shape, name=None, batchSize=None,  **kwargs) :
        super(Input, self).__init__(name=name, **kwargs)
        
        if isinstance(shape, int) :
            sh = (batchSize, shape)
        elif isinstance(shape, float) :
            sh = (batchSize, int(shape))
        else :
            sh = [batchSize]
            sh.extend(list(shape))
            sh = tuple(sh)
        
        self.setHP("shape", sh)
        
        # self.broadcastable = [s == 1 for s in self.shape]
        if len(self.getHP("shape")) == 2:
            typ = tt.matrix
        elif len(self.getHP("shape")) == 3:
            typ = tt.tensor3
        elif len(self.getHP("shape")) == 4:
            typ = tt.tensor4
        elif len(self.getHP("shape")) == 5:
            typ = tt.tensor5
        else :
            raise ValueError("The maximum shape size allowed is 5")

        self.inputs = MTYPES.Inputs(typ, streams=self.streams)
        for f in self.streams :
            self.outputs[f]=self.inputs[f]
    
    def setInputs(self) :
        pass

    def getShape_abs(self) :
        return self.getHP("shape")

    def setOutputs_abs(self) :
        "initializes the output to be the same as the inputs"
        pass
    
    def femaleConnect(self, *args) :
        raise ValueError("Nothing can be connected to an input layer")

class Merge(Layer_ABC):
    """Merge layers using basic arithmetic"""
    def __init__(self, operations, **kwargs):
        super(Merge, self).__init__(maxInConnections=None, **kwargs)
        self.operations = operations
        self.outputs = MTYPES.Variable(streams = self.operations.streams)

        for l in self.operations.getDependencies() :
            l.connect(self)

    def setInputs(self) :
        pass

    def femaleConnect(self, layer) :
        raise ValueError("You can't connect something to a MergeLayer")

    def _femaleConnect(self, layer) :
        pass

    def getShape_abs(self) :
        return self.operations.getShape_abs()

    def setOutputs_abs(self) :
        self.outputs = self.operations.getOutputs()
#Shortcut
M = Merge
# M(a + b * c - o, ...)

class SingleOptMerge(Merge):
    """docstring for SingleOptMerge"""
    def __init__(self, layers, operation, **kwargs):
        operations = ArithmeticMerge(layers[0], layers[1], operation)
        for l in layers[2:] :
            operations = ArithmeticMerge(self.operations, layers[1], operation)
        super(SingleOptMerge, self).__init__(operations, **kwargs)

class Add(SingleOptMerge):
    """docstring for Addition"""
    def __init__(self, layers, **kwargs):
        super(Add, self).__init__(layers, "+", **kwargs)
 
class Multiply(SingleOptMerge):
    """docstring for Multiply"""
    def __init__(self, layers, **kwargs):
        super(Multiply, self).__init__(layers, "*", **kwargs)

class Substract(SingleOptMerge):
    """docstring for Substract"""
    def __init__(self, layers, **kwargs):
        super(Substract, self).__init__(layers, "-", **kwargs)

class Divide(SingleOptMerge):
    """docstring for Divide"""
    def __init__(self, layers, **kwargs):
        super(Divide, self).__init__(layers, "/", **kwargs)

class Concatenate(Layer_ABC):
    """Concatenate layers"""
    def __init__(self, layers, croppings = [], axis=1, **kwargs):
        from lasagne.layers.merge import autocrop_array_shapes
        
        super(Concatenate, self).__init__(maxInConnections=None, **kwargs)
        self.layers = list(layers)
        self.croppings = croppings
        self.axis = axis

        self.croppedShapes = []
        self.originalShapes = []
        for l in self.layers :
            self.originalShapes.append(l.getShape_abs())
        self.croppedShapes = autocrop_array_shapes(self.originalShapes, self.croppings)
        
        self.shape = []
        for i in xrange(len(self.croppedShapes[0])) :
            if i == self.axis :
                v = 0
                for s in self.croppedShapes :
                    v += s[i]
            else :
                v = self.croppedShapes[0][i]
            self.shape.append(v)

        self.shape = tuple(self.shape)

        self.streams = None
        for l in self.layers :
            l.connect(self)
            if self.streams is None :
                self.streams = set(l.outputs.streams)
            else :
                sl = set(l.outputs.streams)
                notCommon = self.streams - sl
                if len(notCommon) > 0 :
                    print MCAN.warning("Layers %s do not share streams: %s with other layers .Will take the intersection" %(l.name, notCommon))
                    self.streams = self.streams & sl

        self.outputs = MTYPES.Variable(streams = self.streams)

    def setInputs(self) :
        pass

    def getShape_abs(self) :
        return self.shape

    def setOutputs_abs(self) :
        from lasagne.layers.merge import autocrop
        for s in self.streams :
            outs = []
            for l in self.layers :
                outs.append(l.outputs[s])
            self.outputs[s] = tt.concatenate(autocrop(outs, self.croppings), axis=self.axis)

#Shortcut
C = Concatenate
# C( [a, b, c], ...)

class Embedding(Layer_ABC) :
    """Embeddings are learned representations of the inputs that are much loved in NLP.
    This layer will take care of creating the embeddings and optimizing them. It can either be used as an input layer or as hidden layer"""

    def __init__(self, nbDimensions, dictSize, name=None, zeroForNull=False, initializations=[MI.Uniform('embeddings', small=True)], **kwargs) :
        """
        :param int nbDimensions: the number of dimensions in wich to encode each word.
        :param int dictSize: the total number of words.
        :param bool zeroForNull: if True the dictionnary will be augmented by one elements at te begining (index=0) whose parameters will always be vector of zeros. This can be used to selectively mask some words in the input, but keep in mind that the index for the first word is moved to one.
        :param int size: the size of the input vector (if your input is a sentence this should be the number of words in it). You don't have to provide a value in the embedding layer is a hidden layer
        
        """

        super(Embedding, self).__init__(initializations=initializations, name=name, **kwargs)

        self.zeroForNull=zeroForNull

        self.setHP("dictSize", dictSize)
        self.setHP("nbDimensions", nbDimensions)

        self.setP("embeddings", MTYPES.Parameter(name="%s.%s" % (self.name, "embeddings")))

        self.inputs=MTYPES.Variable(streams = self.streams)

    def femaleConnect(self, layer) :
        if layer.getDimensionality() != 1 :
            raise ValueError("Input layer must be a vector, got %s dimensions" % layer.getDimensionality())
        self.nbInputs = layer.getShape_abs()[1]

    def getShape_abs(self) :
        return (None, self.nbInputs * self.getHP("nbDimensions"))

    def getParameterShape_abs(self, param, **kwargs) :
        if param == "embeddings" :
            return (self.getHP("dictSize"), self.getHP("nbDimensions"))

    def setInputs(self) :
        inpLayer = list(self.getInLayers())[0]
        if inpLayer.outputs[self.streams[0]].dtype.find("int") != 0 :
            for f in self.streams :
                self.inputs[f] = tt.cast(inpLayer.outputs[f], dtype=MSET.INTX)
        else :
            for f in self.streams :
               self.inputs[f] = inpLayer.outputs[f]
        
    def setOutputs_abs(self) :
        if self.zeroForNull :
            self.null=numpy.zeros((1, self.getHP("nbDimensions")))
            embs = self.parameters["embeddings"].getValue()
            self.parameters["embeddings"].setValue( tt.concatenate( [self.null, embs], axis=0) )
       
        for s in self.streams :
            preOutputs=self.parameters["embeddings"].getVar()[self.inputs[s]]
            self.outputs[s] = preOutputs.reshape((self.inputs[s].shape[0], self.getHP("nbDimensions")))

class Pass(Layer_ABC) :
    def __init__(self, name=None, **kwargs):
        super(Pass, self).__init__(name=name, **kwargs)
    
    def getShape_abs(self) :
        return self.network.getInConnections(self)[0].getShape_abs()

    def setOutputs_abs(self) :
        layer = self.network.getInConnections(self)[0]
        for f in layer.outputs.streams :
            self.outputs[f] = layer.outputs[f]

class Dense(Layer_ABC) :
    """A layer with weigth and bias. If would like to disable either one of them do not provide an initialization"""

    def __init__(self, size, initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)], **kwargs) :
        super(Dense, self).__init__(initializations=initializations, **kwargs)
        if isinstance(size, int) :
            sh = (None, size)
        elif isinstance(size, float) :
            sh = (None, int(size))
        else :
            sh = [None]
            sh.extend(list(size))
            sh = tuple(sh)
            
        self.size = size
        self.setHP("shape", sh)

        self.setParameters({
            "W": MTYPES.Parameter("%s.W" % (self.name)),
            "b": MTYPES.Parameter("%s.b" % (self.name))
        })

        self.inputShape=None
        self.originalInputShape=None

    def setShape_abs(self) :
        # print self.network.getInConnections(self)
        layer = self.network.getInConnections(self)[0]
        if layer.getShape_abs() :
            self.originalInputShape = tuple(layer.getShape_abs())
            if len(self.originalInputShape) > 2 :
                s = 1
                for v in self.originalInputShape[1:] :
                    if v is not None :
                        s *= v
                self.inputShape = (-1, s)
            else :
               self.inputShape = self.originalInputShape

    def femaleConnect(self, layer) :
        self.setShape_abs()

    def getShape_abs(self) :
        """defines the number of inputs"""
        return self.getHP("shape")

    def getParameterShape_abs(self, param, **kwargs) :
        if param == "W" :
            return self.inputShape[1:] + self.getHP("shape")[1:]
        elif param == "b" :
            return self.getHP("shape")[1:]
        else :
            raise ValueError("Unknown parameter: %s" % param)

    def setOutputs_abs(self) :
        """Defines, self.outputs["train"] and self.outputs["test"]"""
        if self.getP("W") is not None:
            for s in self.inputs.streams :
                # print self, self.inputs[s], self.getP("W")()
                self.outputs[s]=tt.dot(self.inputs[s], self.getP("W")())
                if self.getP("b") is not None:
                    self.outputs[s]=self.outputs[s] + self.getP("b")()

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
        super(Output_ABC, self).__init__(**kwargs)
        self.targets=MTYPES.Targets()
        self.dependencies=OrderedDict()
        self.backTrckAll=backTrckAll

        self.abstractions["cost"] = [cost]
        self.loss=None
        self.dependencies=None

    def _backTrckDependencies(self, force=False) :
        """Finds all the hidden layers the output layer is influenced by"""
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

    def _setTheanoFunctions(self) :
        """
        Sets all the theano function.
        """
        super(Output_ABC, self)._setTheanoFunctions()
        self._backTrckDependencies()
        
        self.loss = MTYPES.Losses(self, self.abstractions["cost"][0], self.targets, self.outputs)
        
        self.drive = MWRAP.TheanoFunctionGroup("drive", self, self.loss, allow_input_downcast=True)
        self.drive.allowUpdates("train")

        self.train = self.drive["train"]
        self.test = self.drive["test"]

class DenseOutput_ABC(Output_ABC, Dense):
    """Generic output layer with weight and bias"""
    def __init__(self, size, cost, learningScenari, activation, **kwargs):
        super(DenseOutput_ABC, self).__init__(size=size, cost=cost, learningScenari=learningScenari, activation=activation, **kwargs)

    def setOutputs_abs(self) :
        Dense.setOutputs_abs(self)

class PassOutput_ABC(Output_ABC, Pass):
    """Generic output layer with weight and bias"""
    def __init__(self, cost, learningScenari, activation, **kwargs):
        super(PassOutput_ABC, self).__init__(cost=cost, learningScenari=learningScenari, activation=activation, **kwargs)

class SoftmaxClassifier(DenseOutput_ABC) :
    """A softmax (probabilistic) Classifier"""
    def __init__(self, nbClasses, cost, learningScenari, temperature=1, **kwargs) :
        if not isinstance(nbClasses, int) :
            raise ValueError("nbClasses must be an integer")
        self.nbClasses = nbClasses
        super(SoftmaxClassifier, self).__init__(nbClasses, cost=cost, learningScenari=learningScenari, activation=MA.Softmax(temperature=temperature), **kwargs)
        self.targets=MTYPES.Targets(tt.ivector)

    def _setTheanoFunctions(self) :
        """defines::

            * classify: return the argmax of the outputs applying all the decorators.
            * predict: return the argmax of the test outputs (some decorators may not be applied).
            * classificationAccuracy: returns the accuracy (between [0, 1]) of the model, computed on outputs.
            * predictionAccuracy: returns the accuracy (between [0, 1]) of the model, computed on test outputs.
        """
        super(SoftmaxClassifier, self)._setTheanoFunctions()
        pred= {
            "train": tt.argmax(self.outputs["train"], axis=1),
            "test": tt.argmax(self.outputs["test"], axis=1)
        }
        acc={
            "train": tt.mean( tt.eq(self.targets["train"], pred['train']) ),
            "test": tt.mean( tt.eq(self.targets["test"], pred["test"]) )
        }

        self.predict = MWRAP.TheanoFunctionGroup("predict", self, pred, allow_input_downcast=True, on_unused_input='ignore')
        self.accuracy = MWRAP.TheanoFunctionGroup("accuracy", self, acc, allow_input_downcast=True)

class Regression(DenseOutput_ABC) :
    """For regressions, works great with a mean squared error cost"""
    def __init__(self, size, activation, learningScenari, cost, name=None, **kwargs) :
        super(Regression, self).__init__(size, activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targets=MTYPES.Targets(tt.matrix)

class PassRegression(PassOutput_ABC) :
    """For regressions, works great with a mean squared error cost"""
    def __init__(self, activation, learningScenari, cost, name=None, **kwargs) :
        super(PassRegression, self).__init__(activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targets=MTYPES.Targets(tt.matrix)

class Autoencode(DenseOutput_ABC) :
    """An auto encoding layer. This one takes another layer as inputs and tries to reconstruct its activations.
    You could achieve the same result with a Regression layer, but this one has the advantage of not needing to be fed specific inputs"""

    def __init__(self, targetLayer, activation, learningScenari, cost, name=None, **kwargs) :
        super(Autoencode, self).__init__(targetLayer.getIntrinsicShape(), activation=activation, learningScenari=learningScenari, cost=cost, name=name, **kwargs)
        self.targetLayerName=targetLayer.name
 
    def _whateverFirstInit(self) :
        self.targets.tie(self.network[self.targetLayerName].outputs)
