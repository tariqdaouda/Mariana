import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
from Mariana.abstraction import Abstraction_ABC
import Mariana.initializations as MI

__all__= ["Decorator_ABC", "BatchNormalization", "Center", "Normalize", "Mask", "RandomMask", "BinomialDropout", "Clip", "AdditiveGaussianNoise", "MultiplicativeGaussianNoise"]

def iCast(thing) :
    if thing.dtype.find("int") > -1 :
        return tt.cast(thing, MSET.INTX)
    else :
        return tt.cast(thing, theano.config.floatX)

class Decorator_ABC(Abstraction_ABC) :
    """A decorator is a modifier that is applied on a layer. They are always the last the abstraction to be applied and they can transform a layer in anyway they want."""

    def __call__(self, **kwargs) :
        self.decorate(**kwargs)

    def apply(self, layer) :
        """Apply to a layer and update networks's log"""
        hyps = {}
        for k in self.hyperParameters :
            hyps[k] = getattr(self, k)

        message = "%s is decorated by %s" % (layer.name, self.__class__.__name__)
        layer.network.logLayerEvent(layer, message, hyps)
        return self.decorate(layer)

    def decorate(self, layer) :
        """The function that all decorator_ABCs must implement"""
        raise NotImplemented("This one should be implemented in child")

class Mask(Decorator_ABC):
    """Applies a fixed mask to the outputs of the layer. This layers does this:

        .. math::

            outputs = outputs * mask

    If you want to remove some parts of the output use 0s, if you want to keep them as they are use 1s.
    Anything else will lower or increase the values by the given factors.

    :param array/list mask: It should have the same dimensions as the layer's outputs.

    """
    def __init__(self, mask, onTrain=True, onTest=False):
        Decorator_ABC.__init__(self)
        self.mask = tt.cast(mask, theano.config.floatX)
        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["onTrain", "onTest"])

    def decorate(self, layer) :
        if self.onTrain :
            layer.outputs["train"] = layer.outputs["train"] * self.mask
        
        if self.onTest :
            layer.outputs["test"] = layer.outputs["test"] * self.mask

class RandomMask(Decorator_ABC):
    """
    This decorator takes a list of masks and will randomly apply them to the outputs of the layer it decorates.
    Could be used as a fast approximation for dropout. 
    """
    def __init__(self, masks, onTrain=True, onTest=False, *args, **kwargs):
        Decorator_ABC.__init__(self)
        self.nbMasks = len(masks)
        self.onTest = onTest
        self.onTrain = onTrain

        if self.nbMasks > 0 :
            self.masks = theano.shared(numpy.asarray(masks, dtype=theano.config.floatX))
            self.hyperParameters.extend([])

    def decorate(self, layer) :
        if self.nbMasks > 0 :
            rnd = tt.shared_randomstreams.RandomStreams()
            maskId = rnd.random_integers(low=0, high=self.nbMasks-1, ndim=1)
            mask = self.masks[maskId]

            if self.onTrain :
                layer.outputs["train"] = layer.outputs["train"] * mask
    
            if self.onTest :
                layer.outputs["test"] = layer.outputs["test"] * mask

class BinomialDropout(Decorator_ABC):
    """Stochastically mask some parts of the output of the layer. Use it to make things such as denoising autoencoders and dropout layers"""
    def __init__(self, ratio, onTrain=True, onTest=False):
        Decorator_ABC.__init__(self)

        assert (ratio >= 0 and ratio <= 1)
        self.ratio = ratio
        self.seed = MSET.RANDOM_SEED
        
        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["ratio", "onTrain", "onTest"])

    def _decorate(self, outputs) :
        if self.ratio <= 0 :
            return outputs
        rnd = tt.shared_randomstreams.RandomStreams()
        mask = rnd.binomial(n = 1, p = (1-self.ratio), size = outputs.shape)
        # cast to stay in GPU float limit
        mask = iCast(mask)
        return (outputs * mask)

    def decorate(self, layer) :
        if self.onTrain :
            layer.outputs["train"] = self._decorate(layer.outputs["train"])
        
        if self.onTest :
            layer.outputs["test"] = self._decorate(layer.outputs["test"])

class Center(Decorator_ABC) :
    """Centers the outputs by substracting the mean"""
    def __init__(self, onTrain=True, onTest=True):
        Decorator_ABC.__init__(self)
        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["onTrain", "onTest"])

    def decorate(self, layer) :
        if self.onTrain :
            layer.outputs["train"] = layer.outputs["train"]-tt.mean(layer.outputs["train"])
        if self.onTest :
            layer.outputs["test"] = layer.outputs["test"]-tt.mean(layer.outputs["test"])

class Normalize(Decorator_ABC) :
    """
    Normalizes the outputs by substracting the mean and deviding by the standard deviation

    :param float espilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
    """

    def __init__(self, espilon=1e-6, onTrain=True, onTest=True) :
        Decorator_ABC.__init__(self)
        self.espilon = espilon
        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["onTrain", "onTest", "epsilon"])

    def decorate(self, layer) :
        if self.onTrain :
            std = tt.sqrt( tt.var(layer.outputs["train"]) + self.espilon )
            layer.outputs["train"] = ( layer.outputs["train"]-tt.mean(layer.output) / std )

        if self.onTest :
            std = tt.sqrt( tt.var(layer.outputs["test"]) + self.espilon )
            layer.outputs["test"] = ( layer.outputs["test"]-tt.mean(layer.outputs["test"]) ) / std

class BatchNormalization_deprectaded_to_become_layer(Decorator_ABC):
    """Applies Batch Normalization to the outputs of the layer.
    Implementation according to Sergey Ioffe and Christian Szegedy (http://arxiv.org/abs/1502.03167)
    
        .. math::

            W * ( inputs - mean(mu) )/( std(inputs) ) + b

        Where W and b are learned and std stands for the standard deviation. The mean and the std are computed accross the whole minibatch.

        :param float epsilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
        :param initialization WInitialization: How to initizalise the weights. This decorator is smart enough to use layer initializations.
        :param initialization bInitialization: Same for bias
    """

    def __init__(self, WInitialization=MI.SmallUniformWeights(), bInitialization=MI.ZeroBias(), epsilon=1e-6, onTrain=True, onTest=True) :
        Decorator_ABC.__init__(self)
        self.epsilon = epsilon
        self.WInitialization = WInitialization
        self.bInitialization = bInitialization
        self.W = None
        self.b = None
        self.paramShape = None

        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["onTrain", "onTest", "epsilon"])

    def initParameter(self, parameter, value) :
        setattr(self, parameter, value)

    def getParameterShape(self, **kwargs) :
        return self.paramShape

    def decorate(self, layer) :
        if not hasattr(layer, "batchnorm_W") or not hasattr(layer, "batchnorm_b") :
            self.paramShape = layer.getOutputShape()#(layer.nbOutputs, )
            self.WInitialization.initialize(self)
            self.bInitialization.initialize(self)

            layer.batchnorm_W = self.W
            layer.batchnorm_b = self.b

            if self.onTrain :
                mu = tt.mean(layer.outputs["train"])
                sigma = tt.sqrt( tt.var(layer.outputs["train"]) + self.epsilon )
                layer.outputs["train"] = layer.batchnorm_W * ( (layer.outputs["train"] - mu) / sigma ) + layer.batchnorm_b

            if self.onTest :
                mu = tt.mean(layer.outputs["test"])
                sigma = tt.sqrt( tt.var(layer.outputs["test"]) + self.epsilon )
                layer.outputs["test"] = layer.batchnorm_W * ( (layer.outputs["test"] - mu) / sigma ) + layer.batchnorm_b

class Clip(Decorator_ABC):
    """Clips the neurone activations, preventing them to go beyond the specified range"""
    def __init__(self, lower, upper, onTrain=True, onTest=False):
        assert lower < upper
        Decorator_ABC.__init__(self)

        self.upper = upper
        self.lower = lower
        
        self.onTrain = onTrain
        self.onTest = onTest
        self.hyperParameters.extend(["onTrain", "onTest"])
        
    def decorate(self, layer) :
        if self.onTrain :
            layer.outputs["train"] = layer.outputs["train"].clip(self.lower, self.upper)
        if self.onTest :
            layer.outputs["test"] = layer.outputs["test"].clip(self.lower, self.upper)

class AdditiveGaussianNoise(Decorator_ABC):
    """Add gaussian noise to the output of the layer"""
    
    def __init__(self, std, avg=1, onTrain = True, onTest = False, *args, **kwargs):
        self.std = std
        self.avg = avg
        self.hyperParameters = ["std", "avg"]
        self.onTrain = onTrain
        self.onTest = onTest
        Decorator_ABC.__init__(self, *args, **kwargs)
        
    def _decorate(self, outputs, std) :
        rnd = tt.shared_randomstreams.RandomStreams()
        randomPick = rnd.normal(size = outputs.shape, avg=self.avg, std=std)
        return (outputs + randomPick)
    
    def decorate(self, layer) :
        if self.std > 0 :
            if self.onTrain :
                layer.outputs["train"] = self._decorate(layer.outputs["train"], self.std)
            if self.onTest :
                layer.outputs["test"] = self._decorate(layer.outputs["test"], self.std)

class MultiplicativeGaussianNoise(Decorator_ABC):
    """Multiply gaussian noise to the output of the layer"""
   
    def __init__(self, std, avg=1, onTrain = True, onTest = False, *args, **kwargs):
        self.std = std
        self.avg = avg
        self.hyperParameters = ["std", "avg"]
        self.onTrain = onTrain
        self.onTest = onTest
        Decorator_ABC.__init__(self, *args, **kwargs)
        
    def _decorate(self, outputs, std) :
        rnd = tt.shared_randomstreams.RandomStreams()
        randomPick = rnd.normal(size = outputs.shape, avg=self.avg, std=std)
        return (outputs * randomPick)

    def decorate(self, layer) :
        if self.std > 0 :
            if self.onTrain :
                layer.outputs["train"] = self._decorate(layer.outputs["train"], self.std)
            if self.onTest :
                layer.outputs["test"] = self._decorate(layer.outputs["test"], self.std)