import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
from Mariana.abstraction import Abstraction_ABC

__all__= [
    "Initialization_ABC",
    "HardSet",
    "Identity",
    "GlorotTanhInit",
    "Uniform",
    "UniformWeights",
    "UniformEmbeddings",
    "SmallUniform",
    "SmallUniformWeights",
    "SmallUniformEmbeddings",
    "Normal",
    "NormalWeights",
    "SingleValue",
    "SingleValueWeights",
    "SingleValueBias",
    "ZeroWeights",
    "ZeroBias"
]

class Initialization_ABC(Abstraction_ABC) :
    """This class defines the interface that an Initialization must offer. As a general good practice every init should only take care of a single
    parameter, but you are free to do whatever you want.
    """

    def __call__(self, *args, **kwargs) :
        self.initialize(*args, **kwargs)

    def apply(self, layer) :
        hyps = {}
        for k in self.hyperParameters :
            hyps[k] = getattr(self, k)

        message = "%s was initialized using %s" % (layer.name, self.__class__.__name__)
        try :
            self.initialize(layer)
        except ValueError  as e:
            message = "%s was *NOT* initialized using %s. Because: %s" % (layer.name, self.__class__.__name__, e.message)

        layer.network.logLayerEvent(layer, message, hyps)

    def initialize(self, layer) :
        """The function that all Initialization_ABCs must implement"""
        raise NotImplemented("This one should be implemented in child")

class Identity(Initialization_ABC) :
    """Identity matrix for weights"""
    def __init__(self, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)

    def initialize(self, layer) :
        v = numpy.identity(layer.nbOutputs, dtype = theano.config.floatX)
        layer.initParameter( "W",  theano.shared(value = v, name = "%s_%s" % (layer.name, "W") ) )

class HardSet(Initialization_ABC) :
    """Sets the parameter to value (must have a correct shape)"""
    def __init__(self, parameter, value, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)
        self.parameter = parameter
        self.value = value
        self.hyperParameters = [parameter]

    def initialize(self, layer) :
        layer.initParameter( self.parameter,  theano.shared(value = self.value, name = "%s_%s" % (layer.name, self.parameter) ) )

class GlorotTanhInit(Initialization_ABC) :
    """Set up the layer weights according to the tanh initialization introduced by Glorot et al. 2010"""
    def __init__(self, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)

    def initialize(self, layer) :
        shape = layer.getParameterShape("W")
        rng = numpy.random.RandomState(MSET.RANDOM_SEED)

        W = rng.uniform(
                    low = -numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
                    high = numpy.sqrt(6. / (layer.nbInputs + layer.nbOutputs)),
                    size = shape
                )
        layer.initParameter( "W", theano.shared(W) )

class Uniform(Initialization_ABC) :
    """Random values from a unifrom distribution (divided by the overall sum)."""
    def __init__(self, parameter, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)
        self.parameter = parameter
        self.hyperParameters = [parameter]

    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        v = numpy.random.random(shape)
        v = numpy.asarray(v, dtype=theano.config.floatX)
        layer.initParameter( self.parameter,  theano.shared(value = v, name = "%s_%s" % (layer.name, self.parameter) ) )

class UniformWeights(Uniform) :
    """Small random weights from a unifrom distribution"""
    def __init__(self, *args, **kwargs) :
        Uniform.__init__(self, 'W', *args, **kwargs)
        self.hyperParameters = []

class UniformEmbeddings(Uniform) :
    """Random embeddings from a unifrom distribution"""
    def __init__(self, *args, **kwargs) :
        Uniform.__init__(self, 'embeddings', *args, **kwargs)
        self.hyperParameters = []

class SmallUniform(Uniform) :
    """Random values from a unifrom distribution (divided by the overall sum)."""
    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        try :
            v = numpy.random.random(shape)
        except :
            raise KeyError("Layer '%s' has weird shape: '%s'" % (layer.name, shape))

        v /= sum(v)
        v = numpy.asarray(v, dtype=theano.config.floatX)
        layer.initParameter( self.parameter,  theano.shared(value = v, name = "%s_%s" % (layer.name, self.parameter)) )

class SmallUniformWeights(SmallUniform) :
    """Small random weights from a unifrom distribution (divided by the overall sum)"""
    def __init__(self, *args, **kwargs) :
        SmallUniform.__init__(self, 'W', *args, **kwargs)
        self.hyperParameters = []

class SmallUniformEmbeddings(SmallUniform) :
    """Small random embeddings from a unifrom distribution (divided by the overall sum)"""
    def __init__(self, *args, **kwargs) :
        SmallUniform.__init__(self, 'embeddings', *args, **kwargs)
        self.hyperParameters = []

class ScaledVarianceWeights(Initialization_ABC):
    """Scales the weights so that the variance is the same for every neurone."""
    def __init__(self, *args, **kwargs):
        self.parameter = "W"
        super(UnitVariance, self).__init__(*args, **kwargs)
    
    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        w = numpy.random.randn(shape[0]) / numpy.sqrt(shape[0])
        layer.initParameter( "W",  theano.shared(value = w, name = "%s_%s" % (layer.name, "W") ) )

class Normal(Initialization_ABC) :
    """Random values from a normal distribution"""
    def __init__(self, parameter, standardDev, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)
        self.parameter = parameter
        self.standardDev = standardDev
        self.hyperParameters = ["parameter", "standardDev"]

    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        v = numpy.random.normal(0, self.standardDev, shape)
        v = numpy.asarray(v, dtype=theano.config.floatX)
        layer.initParameter( self.parameter,  theano.shared(value = v, name = "%s_%s" % (layer.name, self.parameter) ) )

class NormalWeights(Normal) :
    """Random weights from a normal distribution"""
    def __init__(self, standardDev, *args, **kwargs) :
        Normal.__init__(self, 'W', standardDev, *args, **kwargs)
        self.standardDev = standardDev
        self.hyperParameters = ["standardDev"]

class HeWeights(Initialization_ABC) :
    """Initialization proposed by He et al. for ReLU in *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification* """
    def __init__(self, *args, **kwargs) :
        super(HeWeights, self).__init__(*args, **kwargs)
        self.parameter = "W"
        self.hyperParameters = []

    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        v = numpy.random.normal(0, numpy.sqrt(2./shape[0]), shape)
        layer.initParameter( self.parameter,  theano.shared(value = v, name = "%s_%s" % (layer.name, self.parameter) ) )

class SingleValue(Initialization_ABC) :
    """Initialize to a given value"""
    def __init__(self, parameter, value, *args, **kwargs) :
        Initialization_ABC.__init__(self, *args, **kwargs)
        self.parameter = parameter
        self.value = value
        self.hyperParameters = ["parameter", "value"]

    def initialize(self, layer) :
        shape = layer.getParameterShape(self.parameter)
        v = numpy.zeros( shape, dtype = theano.config.floatX) + self.value
        layer.initParameter( self.parameter,  theano.shared(value = v, name = "%s_%s" % (layer.name, self.parameter) ) )

class SingleValueWeights(SingleValue) :
    """Initialize the weights to a given value"""
    def __init__(self, value, *args, **kwargs) :
        SingleValue.__init__(self, 'W', value, *args, **kwargs)
        self.hyperParameters = ["value"]

class ZeroWeights(SingleValueWeights) :
    """Initialize the weights to zero"""
    def __init__(self, *args, **kwargs) :
        SingleValueWeights.__init__(self, 0)
        self.hyperParameters = []

class SingleValueBias(SingleValue) :
    """Initialize the bias to a given value"""
    def __init__(self, value, *args, **kwargs) :
        SingleValue.__init__(self, 'b', value, *args, **kwargs)
        self.hyperParameters = ["value"]

class ZeroBias(SingleValueBias) :
    """Initialize the bias to zeros"""
    def __init__(self, *args, **kwargs) :
        SingleValueBias.__init__(self, 0)
        self.hyperParameters = []
