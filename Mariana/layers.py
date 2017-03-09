from abc import ABCMeta#, abstractmethod
from collections import OrderedDict

# import types
import theano, numpy, time
import theano.tensor as tt

import Mariana.activations as MA
import Mariana.initializations as MI
import Mariana.settings as MSET

import Mariana.network as MNET
import Mariana.wrappers as MWRAP
import Mariana.candies as MCAN

__all__=["Layer_ABC", "WeightBiasOutput_ABC", "WeightBias_ABC", "Output_ABC", "Input", "Hidden", "Composite", "Embedding", "SoftmaxClassifier", "Regression", "Autoencode"]

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
        learningScenario=None,
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
        self.inputs=None
        self.nbOutputs=size
        self.outputs=None # this is a symbolic var
        self.testOutputs=None # this is a symbolic var

        self.preactivation_outputs=None
        self.preactivation_testOutputs=None

        self.activation=activation
        self.regularizationObjects=regularizations
        self.regularizations=[]
        self.decorators=decorators
        self.initializations=initializations
        self.learningScenario=learningScenario

        self.network=MNET.Network()
        self.network._addLayer(self)

        self._inputRegistrations=set()

        self._mustInit=True
        self._mustReset=True
        self._decorating=False

        self.parameters = {}

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
            for init in self.initializations :
                init.apply(self)
        self._mustReset=False

    def initParameter(self, parameter, value) :
        """Initialize a parameter, raise value error if already initialized"""
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
    def _setOutputs(self) :
        """Defines the outputs and testOutputs of the layer before the application of the activation function. This function is called by _init() ans should be written in child."""
        raise NotImplemented("Should be implemented in child")

    def _decorate(self) :
        """applies decorators"""
        for d in self.decorators :
            d.apply(self)

    def _activate(self) :
        """applies activation"""
        self.preactivation_outputs=self.outputs
        self.preactivation_testOutputs=self.testOutputs

        self.outputs=self.activation.apply(self, self.preactivation_outputs, 'training')
        self.testOutputs=self.activation.apply(self, self.preactivation_testOutputs, 'testing')

    def _listRegularizations(self) :
        self.regularizations=[]
        for reg in self.regularizationObjects :
            self.regularizations.append(reg.getFormula(self))

    def _setTheanoFunctions(self) :
        """Creates propagate/propagateTest theano function that returns the layer's outputs.
        propagateTest returns the testOutputs, some decorators might not be applied.
        This is called after decorating"""
        self.propagate=MWRAP.TheanoFunction("propagate", self, [("outputs", self.outputs)], allow_input_downcast=True)
        self.propagateTest=MWRAP.TheanoFunction("propagateTest", self, [("outputs", self.testOutputs)], allow_input_downcast=True)
        
        # self.propagate_preAct=MWRAP.TheanoFunction("propagate_preAct", self, [("outputs", self.preactivation_outputs)], allow_input_downcast=True)
        # self.propagateTest_preAct=MWRAP.TheanoFunction("propagateTest_preAct", self, [("outputs", self.preactivation_testOutputs)], allow_input_downcast=True)

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
            if self.testOutputs is None :
                raise AttributeError("Attribute 'testOutputs' of layer '%s' has None value. This attribute defines the test output of the layer, usually without regularizations" % self.name)
        except AttributeError :
                raise AttributeError("Attribute 'testOutputs' of layer '%s' is not defined. This attribute defines the test output of the layer, usually without regularizations" % self.name)

        try :
            if self.outputs is None :
                raise AttributeError("Attribute 'outputs' of layer '%s' has None value. This attribute defines the train output of the layer, usually with regularizations" % self.name)
        except AttributeError :
                raise AttributeError("Attribute 'outputs' of layer '%s' is not defined. This attribute defines the train output of the layer, usually with regularizations" % self.name)

    def _initA(self) :
        """Initialize the essential attributes of the layer such as: outputs and activations. This function is automatically called before train/test etc..."""
        if ( self._mustInit ) and ( len(self._inputRegistrations) == len(self.network.inConnections[self]) ) :
            self._whateverFirstInit()
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
        self._listRegularizations()
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
        self.inputs=tt.matrix(name="inp_"+self.name)

    def _setOutputs(self) :
        "initializes the output to be the same as the inputs"
        self.outputs=self.inputs
        self.testOutputs=self.inputs

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

        self.inputs=None
        self.testInputs=None
        
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
            if self.inputs is None :
                self.inputs=tt.imatrix(name="embInp_" + self.name)
                self.testInputs=self.inputs
        else :
            for layer in self.network.inConnections[self] :
                if layer.outputs.dtype.find("int") != 0 :
                    outs=tt.cast(layer.outputs, dtype=MSET.INTX)
                    testOuts=tt.cast(layer.testOutputs, dtype=MSET.INTX)
                else :
                    outs=layer.outputs
                    testOuts=layer.testOutputs

                if self.inputs is None :   
                    self.inputs=outs
                    self.testInputs=testOuts
                else :
                    self.inputs+=outs
                    self.testInputs+=testOuts

        if self.zeroForNull :
            self.null=numpy.zeros((1, self.nbDimentions))
            self.parameters["fullEmbeddings"]=tt.concatenate( [self.null, self.parameters["embeddings"]], axis=0)
        else :
            self.parameters["fullEmbeddings"]=self.parameters["embeddings"]
            del(self.parameters["embeddings"])

        self.preOutputs=self.parameters["fullEmbeddings"][self.inputs]
        self.outputs=self.preOutputs.reshape((self.inputs.shape[0], self.nbOutputs))
        self.testOutputs=self.preOutputs.reshape((self.testInputs.shape[0], self.nbOutputs))

class Composite(Layer_ABC):
    """A Composite layer concatenates the outputs of several other layers
    for example is we have::

        c=Composite()
        layer1 > c
        layer2 > c

    The output of c will be single vector: [layer1.output, layer2.output]
    """
    def __init__(self, name=None, **kwargs):
        super(Composite, self).__init__(layerTypes=[MSET.TYPE_HIDDEN_LAYER], size=None, name=name, **kwargs)

    def _setShape(self) :
        """set the number of inputs and outputs"""
        self.nbInputs=0
        for l in self.network.inConnections[self] :
            self.nbInputs += l.nbOutputs
        self.nbOutputs=self.nbInputs

    def _setOutputs(self) :
        outs=[]
        for l in self.network.inConnections[self] :
            outs.append(l.outputs)
        
        self.outputs=tt.concatenate( outs, axis=1 )
        self.testOutputs=tt.concatenate( outs, axis=1 )

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
            if self.inputs is None :
                self.inputs=layer.outputs
            else :
                self.inputs += layer.outputs

        self.outputs=self.inputs
        self.testOutputs=self.inputs

class WeightBias_ABC(Layer_ABC) :
    """A layer with weigth and bias. If would like to disable either one of them do not provide an initialization"""

    def __init__(self, size, layerTypes, initializations=[MI.SmallUniformWeights(), MI.ZeroBias()], **kwargs) :
        super(WeightBias_ABC, self).__init__(size, layerTypes=layerTypes, initializations=initializations, **kwargs)
        self.testInputs=None
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
            elif self.nbInputs != layer.nbOutputs :
                raise ValueError("All inputs to layer %s must have the same size, got: %s previous: %s" % (self.name, layer.nbOutputs, self.nbInputs) )

    def _setInputs(self) :
        """Adds up the outputs of all incoming layers"""
        self.inputs=None
        self.testInputs=None
        for layer in self.network.inConnections[self] :
            if self.inputs is None :
                self.inputs=layer.outputs
                self.testInputs=layer.testOutputs
            else :
                self.inputs += layer.outputs
                self.testInputs += layer.testOutputs

    def _setOutputs(self) :
        """Defines, self.outputs and self.testOutputs"""
        self._setInputs()

        self.outputs=self.inputs
        self.testOutputs=self.testInputs
        
        if self.parameters["W"] is not None:
            self.outputs=tt.dot(self.inputs, self.parameters["W"])
            self.testOutputs=tt.dot(self.testInputs, self.parameters["W"])
            
        if self.parameters["b"] is not None:
            self.outputs=self.outputs + self.parameters["b"]
            self.testOutputs=self.testOutputs + self.parameters["b"]

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

    def __init__(self, size, costObject, backTrckAll=False, **kwargs) :
        super(Output_ABC, self).__init__(size, layerTypes=[MSET.TYPE_OUTPUT_LAYER], **kwargs)
        self.targets=None
        self.dependencies=OrderedDict()
        self.costObject=costObject
        self.backTrckAll=backTrckAll

        self.cost=None
        self.testCost=None
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

    def _setCosts(self) :
        """
        Defines the costs to be applied.
        There are 2: the training cost (self.cost) and test cost (self.testCost), that has no regularizations.
         """
        # print "training"
        self.cost=self.costObject.apply(self, self.targets, self.outputs, "training")
        # theano.printing.debugprint(self.cost)
        # print "testing"
        self.testCost=self.costObject.apply(self, self.targets, self.testOutputs, "testing")
        # theano.printing.debugprint(self.testCost)

    def _applyRegularizations(self, force=False) :
        """Defines the regularizations to be added to the cost"""
        if self._mustRegularize or force :
            self._backTrckDependencies()
            for l in self.dependencies.itervalues() :
                if l.__class__  is not Composite :
                    try :
                        for reg in l.regularizations :
                            self.cost += reg
                    except AttributeError :
                        pass
        self._mustRegularize=False

    def _setUpdates(self) :
        """Defines parameter updates according to training scenari"""
        self._backTrckDependencies()
        self.dctUpdates={}
        self.updates=self.learningScenario.apply(self, self.cost)
        self.dctUpdates[self.name]=self.updates
        
        # print self.name, self.updates 
        for l in self.dependencies.itervalues() :
            if l.learningScenario is not None :
                updates=l.learningScenario.apply(l, self.cost)
            else :
                updates=self.learningScenario.apply(l, self.cost)
            # print l.name, updates 
            self.updates.extend(updates)
            self.dctUpdates[l.name]=updates

    def _setTheanoFunctions(self) :
        """
        Sets all the theano function.
        Calls self._setCosts() before if either self.cost or self.testCost is None.
        self._applyRegularizations()
        Calls self._setUpdates() if self.updates is None.
        """
        if self.cost is None or self.testCost is None :
            self._setCosts()

        self._applyRegularizations()
        
        if self.updates is None :
            self._setUpdates()

        super(Output_ABC, self)._setTheanoFunctions()
        self._setCustomTheanoFunctions()
        self._setGetGradientsUpdatesFunctions()
    
    def _setCustomTheanoFunctions(self) :
        """Adds train, test, model functions::

            * train: update parameters and return cost
            * test: do not update parameters and return cost without adding regularizations
        """

        if self.cost is None or self.testCost is None :
            self._setCosts()
        # theano.printing.debugprint(self.cost)
        self.train=MWRAP.TheanoFunction("train", self, [("score", self.cost)], { "targets" : self.targets }, updates=self.updates, allow_input_downcast=True)
        self.test=MWRAP.TheanoFunction("test", self, [("score", self.testCost)], { "targets" : self.targets }, allow_input_downcast=True)

    def _setGetGradientsUpdatesFunctions(self) :
        """Defines functions for retreving gradients/updates"""
        layers=[self]
        layers.extend(self.dependencies.values())
        
        for l in layers :
            try :
                gradOuts=[]
                upsOuts=[]
                for k, v in l.getParameterDict().iteritems() :
                    if l.learningScenario is not None :
                        gradOuts.append( (k, l.learningScenario.gradients[v]) )
                        upsOuts.append( (k, l.learningScenario.updates[v]) )
                    else :
                        gradOuts.append( (k, self.learningScenario.gradients[v]) )
                        upsOuts.append( (k, self.learningScenario.updates[v]) )

                setattr(self, "getGradients_%s" % l.name, MWRAP.TheanoFunction("getGradients", self, gradOuts, { "targets" : self.targets }, allow_input_downcast=True, on_unused_input='ignore') )
                setattr(self, "getUpdates_%s" % l.name, MWRAP.TheanoFunction("getUpdates", self, gradOuts, { "targets" : self.targets }, allow_input_downcast=True, on_unused_input='ignore') )
            except :
                msg="Warning! Unable to setup theano function for retreiving updates and gradients for layer '%s'. Perhaps the current learning scenario is not keeping them stored." % l.name
                self.network.logLayerEvent(self, msg, {})
                if MSET.VERBOSE :
                    print(msg)

    def getGradients(self, layerName=None, *args, **kwargs) :
        if layerName is None :
            lname=self.name
        else :
            lname=layerName

        try :
            return getattr(self, "getGradients_%s" % lname)(*args, **kwargs)
        except AttributeError :
            print("There's no theano function for retreiving gradients for layer '%s'. Perhaps the current learning scenario is not keeping them stored." % layerName)

    def getUpdates(self, layerName=None, *args, **kwargs) :
        if layerName is None :
            lname=self.name
        else :
            lname=layerName

        try :
            return getattr(self, "getUpdates_%s" % lname)(*args, **kwargs)
        except AttributeError :
            print("There's no theano function for retreiving updates for layer '%s'. Perhaps the current learning scenario is not keeping them stored." % layerName)

class WeightBiasOutput_ABC(Output_ABC, WeightBias_ABC):
    """Generic output layer with weight and bias"""
    def __init__(self, size, costObject, learningScenario, activation, **kwargs):
        super(WeightBiasOutput_ABC, self).__init__(size=size, costObject=costObject, learningScenario=learningScenario, activation=activation, **kwargs)

    def _setOutputs(self) :
        WeightBias_ABC._setOutputs(self)
 
class SoftmaxClassifier(WeightBiasOutput_ABC) :
    """A softmax (probabilistic) Classifier"""
    def __init__(self, size, costObject, learningScenario, temperature=1, **kwargs) :
        super(SoftmaxClassifier, self).__init__(size, costObject=costObject, learningScenario=learningScenario, activation=MA.Softmax(temperature=temperature), **kwargs)
        self.targets=tt.ivector(name="targets_" + self.name)

    def _setCustomTheanoFunctions(self) :
        """defines::

            * classify: return the argmax of the outputs applying all the decorators.
            * predict: return the argmax of the test outputs (some decorators may not be applied).
            * classificationAccuracy: returns the accuracy (between [0, 1]) of the model, computed on outputs.
            * predictionAccuracy: returns the accuracy (between [0, 1]) of the model, computed on test outputs.
        """
        Output_ABC._setCustomTheanoFunctions(self)
        clas=tt.argmax(self.outputs, axis=1)
        pred=tt.argmax(self.testOutputs, axis=1)

        self.classify=MWRAP.TheanoFunction("classify", self, [ ("class", clas) ], allow_input_downcast=True)
        self.predict=MWRAP.TheanoFunction("predict", self, [ ("class", pred) ], allow_input_downcast=True)

        clasAcc=tt.mean( tt.eq(self.targets, clas ) )
        predAcc=tt.mean( tt.eq(self.targets, pred ) )

        self.classificationAccuracy=MWRAP.TheanoFunction("classificationAccuracy", self, [("accuracy", clasAcc)], { "targets" : self.targets }, allow_input_downcast=True)
        self.predictionAccuracy=MWRAP.TheanoFunction("predictionAccuracy", self, [("accuracy", predAcc)], { "targets" : self.targets }, allow_input_downcast=True)

        self.trainAndAccuracy=MWRAP.TheanoFunction("trainAndAccuracy", self, [("score", self.cost), ("accuracy", clasAcc)], { "targets" : self.targets },  updates=self.updates, allow_input_downcast=True)
        self.testAndAccuracy=MWRAP.TheanoFunction("testAndAccuracy", self, [("score", self.testCost), ("accuracy", predAcc)], { "targets" : self.targets }, allow_input_downcast=True)

class Regression(WeightBiasOutput_ABC) :
    """For regressions, works great with a mean squared error cost"""
    def __init__(self, size, activation, learningScenario, costObject, name=None, **kwargs) :
        super(Regression, self).__init__(size, activation=activation, learningScenario=learningScenario, costObject=costObject, name=name, **kwargs)
        self.targets=tt.matrix(name="targets")

class Autoencode(WeightBiasOutput_ABC) :
    """An auto encoding layer. This one takes another layer as inputs and tries to reconstruct its activations.
    You could achieve the same result with a Regression layer, but this one has the advantage of not needing to be fed specific inputs"""

    def __init__(self, targetLayerName, activation, learningScenario, costObject, name=None, **kwargs) :
        super(Autoencode, self).__init__(None, activation=activation, learningScenario=learningScenario, costObject=costObject, name=name, **kwargs)
        self.targetLayerName=targetLayerName

    def _setNbOutputs(self) :
        self.nbOutputs=self.network[self.targetLayerName].nbOutputs
        
    def _initParameters(self, forceReset=False) :
        self._setNbOutputs()
        super(Autoencode, self)._initParameters(forceReset)

    def _whateverFirstInit(self) :
        self.targets=self.network[self.targetLayerName].outputs
    
    def _setCustomTheanoFunctions(self) :
        super(Autoencode, self)._setCustomTheanoFunctions()
        self.train=MWRAP.TheanoFunction("train", self, [("score", self.cost)], {}, updates=self.updates, allow_input_downcast=True)
        self.test=MWRAP.TheanoFunction("test", self, [("score", self.testCost)], {}, allow_input_downcast=True)
