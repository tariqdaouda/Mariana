import Mariana.settings as MSET
import Mariana.useful as MUSE
import Mariana.abstraction as MABS
import Mariana.activations as MA

import lasagne.init as LI

import numpy
import theano
import theano.tensor as tt

__all__= [
    "Initialization_ABC",
    "Identity",
    "HardSet",
    "SingleValue",
    "Normal",
    "Uniform",
    "FanInFanOut_ABC",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
]

class Initialization_ABC(MABS.UntrainableAbstraction_ABC, MABS.Apply_ABC) :
    """This class defines the interface that an Initialization must offer.
    
    :param string parameter: the name of the parameter to be initialized
    :param float parameter: how sparse should the result be. 0 => no sparsity, 1 => a matrix of zeros
    """

    def __init__(self, parameter, sparsity=0., **kwargs):
        super(Initialization_ABC, self).__init__(**kwargs)
        
        self.addHyperParameters({
            "parameter": parameter,
            "sparsity": sparsity
        })

    # def _apply(self, abstraction, **kwargs) :
    #     if not abstraction.hasP(self.getHP("parameter")) :
    #         raise ValueError ("'%s' does not have a parameter '%s'" % (abstraction, self.getHP("parameter") ) )
        
    #     if not abstraction.getP(self.getHP("parameter")).isTied() :
    #         super(Initialization_ABC, self)._apply(abstraction, **kwargs)
            
    def logApply(self, layer, **kwargs) :
        message = "Applying '%s' on parameter: '%s' of layer '%s'" % (self.name, self.getHP('parameter'), layer.name)
        self.logEvent(message)

    def apply(self, abstraction, **kwargs) :
        
        retShape = abstraction._getParameterShape_abs(self.getHP("parameter"))

        v = MUSE.iCast_numpy(self.run(retShape))
        if (v.shape != retShape) :
            raise ValueError("Initialization has a wrong shape: %s, parameter shape is: %s " % (v.shape, retShape))
        
        v = MUSE.sparsify(v, self.getHP("sparsity"))
        abstraction.setP(self.getHP("parameter"), v)
    
    def run(self, shape) :
        """The function that all Initialization_ABCs must implement"""
        raise NotImplemented("This one should be implemented in child")

class Null(Initialization_ABC) :
    """Return None, mainly for some cases of lasagne compatibility"""
    def run(self, shape) :
        return None

class Identity(Initialization_ABC) :
    """Identity matrix. Its your job to make sure that the parameter is a square matrix"""
    def run(self, shape) :
        sv = None
        for s in shape :
            if not sv :
                sv = s
            elif sv != s :
                raise ValueError("Shape must be square, got: %s" % shape)

        v = numpy.identity(shape, dtype = theano.config.floatX)
        return v

class HardSet(Initialization_ABC) :
    """Sets the parameter to value. It's your job to make sure that the shape is correct"""
    def __init__(self, parameter, value, **kwargs) :
        super(HardSet, self).__init__(**kwargs)
        self.value = numpy.asarray(value, dtype=theano.config.floatX)
        
    def run(self, shape) :
        return self.value

class SingleValue(Initialization_ABC) :
    """Initialize to a given value"""
    def __init__(self, parameter, value, **kwargs) :
        super(SingleValue, self).__init__(parameter, **kwargs)
        self.setHP("value", value)
    
    def run(self, shape) :
        return numpy.ones(shape) * self.getHP("value")
        
class Normal(Initialization_ABC):
    """
    Initializes using a random normal distribution.
    **Small** uses my personal initialization than I find works very well in most cases with a uniform distribution, simply divides by the sum of the weights.
    """
    def __init__(self, parameter, std, mean, small=False, **kwargs):
        super(Normal, self).__init__(parameter, **kwargs)
        self.addHyperParameters({
            "std": std,
            "mean": mean,
            "small": small
        })
    
    def run(self, shape) :
        v = numpy.random.normal(self.mean, self.std, size=shape)
        if self.small :
            return v / numpy.sum(v)
        return v

class Uniform(Initialization_ABC):
    """
    Initializes using a uniform distribution
    **Small** uses my personal initialization than I find can work very well, simply divides by the sum of the weights.
    """
    def __init__(self, parameter, low=0, high=1, small=False, **kwargs):
        super(Uniform, self).__init__(parameter, **kwargs)
        self.setHP("low", low)
        self.setHP("high", high)
        self.setHP("small", small)
    
    def run(self, shape) :
        v = numpy.random.uniform(high=self.getHP("high"), low=self.getHP("low"), size=shape)
        if self.getHP("small") :
            return v / sum(v)
        return v

class FanInFanOut_ABC(Initialization_ABC) :
    """
    Abtract class for fan_in/_out inits (Glorot and He)
    Over the time people have introduced
    ways to make it work with other various activation functions by modifying a gain factor.
    You can force the gain using the *forceGain* argument, otherwise Mariana will choose
    one for you depending on the abstraction's activation.

        * ReLU: sqrt(2)
        
        * LeakyReLU: sqrt(2/(1+alpha**2)) where alpha is the leakiness

        * Everything else : 1.0
    
    This is an abtract class: see *GlorotNormal*, *GlorotUniform*
    """
    def __init__(self, parameter, forceGain=None, **kwargs) :
        super(FanInFanOut_ABC, self).__init__(parameter, **kwargs)
        self.setHP("forceGain", forceGain)
        self.gain = None

    @classmethod
    def _getGain(cls, activation) :
        """returns the gain with respesct to an activation function"""
        if activation.__class__ is MA.ReLU :
            if activation.leakiness == 0 :
                return numpy.sqrt(2)
            else :
                return numpy.sqrt(2/(1+activation.leakiness**2))
        return 1.0

    def setup(self, abstraction) :
        self.gain = self._getGain(abstraction.abstractions["activation"])

    def apply(self, abstraction) :
        import Mariana.activations as MA

        forceGain = self.getHP("forceGain")
        if forceGain :
            self.gain = forceGain
        else :
            self.gain = self._getGain(abstraction.abstractions["activation"])
        
        return super(FanInFanOut_ABC, self).apply(abstraction)

class XNormal(FanInFanOut_ABC) :
    """
    Initialization strategy introduced by Glorot et al. 2010 on a Normal distribution.
    Uses lasagne as backend.
    """ 
    def run(self, shape) :
        return LI.GlorotNormal(gain = self.gain).sample(shape)

XavierNormal = XNormal
GlorotNormal = XNormal

class XUniform(FanInFanOut_ABC) :
    """
    Initialization strategy introduced by Glorot et al. 2010 on a Uniform distribution.
    If you use tanh() as activation try this one first.
    Uses lasagne as backend.
    """ 
    def run(self, shape) :
        return LI.GlorotUniform(gain = self.gain).sample(shape)

XavierUniform = XUniform
GlorotUniform = XUniform

class HeNormal(FanInFanOut_ABC) :
    """
    Initialization proposed by He et al. for ReLU in *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, 2015.
    
    On a Normal distribution, Uses lasagne as backend.
    """ 
    def run(self, shape) :
        return LI.HeNormal(gain = self.gain).sample(shape)

class HeUniform(FanInFanOut_ABC) :
    """
    Initialization proposed by He et al. for ReLU in *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, 2015.
    
    On a Uniform distribution, Uses lasagne as backend.
    """ 
    def run(self, shape) :
        return LI.HeUniform(gain = self.gain).sample(shape)