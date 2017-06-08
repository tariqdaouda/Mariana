import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
import Mariana.abstraction as MABS
import Mariana.initializations as MI
import Mariana.useful as MUSE

__all__= ["Decorator_ABC", "BatchNormalization", "Center", "Normalize", "Mask", "RandomMask", "BinomialDropout", "Clip", "AdditiveGaussianNoise", "MultiplicativeGaussianNoise"]

class Decorator_ABC(MABS.ApplyAbstraction_ABC) :
    """A decorator is a modifier that is applied on a layer's output. They are always the last the abstraction to be applied."""

    def __init__(self, streams, **kwargs):
        super(Decorator_ABC, self).__init__(**kwargs)
        self.mask = tt.cast(mask, theano.config.floatX)
        self.streams = set(self.streams)
        self.setHP("streams", streams)

    def apply(self, layer, stream) :
        """Apply to a layer and update networks's log"""
        
        message = "%s is decorated by %s on stream %s" % (layer.name, self.__class__.__name__, stream)
        layer.network.logLayerEvent(layer, message, self.hyperParameters)
        return self.run(layer)

    def run(self, layer, stream) :
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
    def __init__(self, mask, streams=["train"], **kwargs):
        super(Mask, self).__init__(streams, **kwargs)
        self.mask = tt.cast(mask, theano.config.floatX)
        self.setHP("mask", mask)
        
    def run(self, layer, stream) :
        if stream in self.streams :
            layer.outputs[stream] = layer.outputs[stream] * self.mask

class RandomMask(Decorator_ABC):
    """
    This decorator takes a list of masks and will randomly apply them to the outputs of the layer it runs.
    Could be used as a fast approximation for dropout. 
    """
    def __init__(self, masks, streams=["train"], **kwargs):
        super(RandomMask, self).__init__(streams, **kwargs)
        self.masks = tt.cast(mask, theano.config.floatX)
        self.setHP("masks", masks)

    def run(self, layer, stream) :
        if stream in self.streams :
            rnd = tt.shared_randomstreams.RandomStreams()
            maskId = rnd.random_integers(low=0, high=self.nbMasks-1, ndim=1)
            mask = self.masks[maskId]

            layer.outputs[stream] = layer.outputs[stream] * mask

class BinomialDropout(Decorator_ABC):
    """Stochastically mask some parts of the output of the layer. Use it to make things such as denoising autoencoders and dropout layers"""
    def __init__(self, ratio, streams=["train"], **kwargs):
        super(BinomialDropout, self).__init__(streams, **kwargs)
        assert (ratio >= 0 and ratio <= 1)
        self.ratio = ratio
        self.seed = MSET.RANDOM_SEED
        self.setHP("ratio", ratio)
        
    def run(self, layer, stream) :        
        if stream in self.streams :
            rnd = tt.shared_randomstreams.RandomStreams()
            mask = rnd.binomial(n = 1, p = (1-self.ratio), size = outputs.shape)
            # cast to stay in GPU float limit
            mask = MUSE.iCast_theano(mask)
            return (layer.outputs[stream] * mask)

class Center(Decorator_ABC) :
    """Centers the outputs by substracting the mean"""
    def __init__(self, streams=["train"], **kwargs):
        super(Center, self).__init__(streams, **kwargs)
        
    def run(self, layer, stream) :
        if stream in self.streams :
            layer.outputs[stream] = layer.outputs[stream]-tt.mean(layer.outputs[stream])

class Normalize(Decorator_ABC) :
    """
    Normalizes the outputs by substracting the mean and deviding by the standard deviation

    :param float espilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
    """

    def __init__(self, espilon=1e-6, streams=["train"]) :
        super(Normalize, self).__init__(streams, **kwargs) 
        self.setHP("espilon", espilon)

    def run(self, layer, stream) :
        if stream in self.streams :
            std = tt.sqrt( tt.var(layer.outputs[stream]) + self.espilon )
            layer.outputs[stream] = ( layer.outputs[stream]-tt.mean(layer.output) / std )

# class BatchNormalization(Decorator_ABC):
#     """Applies Batch Normalization to the outputs of the layer.
#     Implementation according to Sergey Ioffe and Christian Szegedy (http://arxiv.org/abs/1502.03167)
    
#         .. math::

#            \\gamma * \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta

#         Where W and b are learned and std stands for the standard deviation. The mean and the std are computed accross the whole minibatch.

#         :param float epsilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
#         :param initialization GammaInit: How to initizalise the weights. This decorator is smart enough to use layer initializations.
#         :param initialization betaInit: Same for bias
#     """

#     def __init__(self, testMu, testStd, gInit=1, b=0, epsilon=1e-6, trainstreams=["train", "test"]) :
#         super(Normalize, self).__init__(streams, **kwargs)
#         self.setHP("gInit", gInit)
#         self.setHP("bInit", bInit)
#         self.setHP("espilon", espilon)
        
#         self.addParameters({
#             "g": MTYPES.Parameter("batchnorm.g"),
#             "b": MTYPES.Parameter("batchnorm.b")
#         })

#     def initialize(self, layer) :
#         w = numpy.ones(shape) * self.gInit
#         self.parameters[W].setValue(v)
#         b = numpy.ones(shape) * self.bInit
#         self.parameters[W].setValue(b)

#     def initParameter(self, parameter, value) :
#         setattr(self, parameter, value)

#     def getParameterShape_abs(self, **kwargs) :
#         return self.paramShape

#     def run(self, layer, stream) :
#         if stream == "train" :

#         if not hasattr(layer, "batchnorm_W") or not hasattr(layer, "batchnorm_b") :
#             self.paramShape = layer.getOutputShape()#(layer.nbOutputs, )
#             self.WInitialization.initialize(self)
#             self.bInitialization.initialize(self)

#             layer.batchnorm_W = self.W
#             layer.batchnorm_b = self.b

#             if self.onTrain :
#                 mu = tt.mean(layer.outputs["train"])
#                 sigma = tt.sqrt( tt.var(layer.outputs["train"]) + self.epsilon )
#                 layer.outputs["train"] = layer.batchnorm_W * ( (layer.outputs["train"] - mu) / sigma ) + layer.batchnorm_b

#             if self.onTest :
#                 mu = tt.mean(layer.outputs["test"])
#                 sigma = tt.sqrt( tt.var(layer.outputs["test"]) + self.epsilon )
#                 layer.outputs["test"] = layer.batchnorm_W * ( (layer.outputs["test"] - mu) / sigma ) + layer.batchnorm_b

class Clip(Decorator_ABC):
    """Clips the neurone activations, preventing them to go beyond the specified range"""
    
    def __init__(self, lower, upper, streams=["train"] **kwargs) :
        super(Clip, self).__init__(streams, **kwargs) 
        assert lower < upper
        self.setHP("lower", lower)
        self.setHP("upper", upper)
    
    def run(self, layer, stream) :
        if stream in self.streams :
            layer.outputs[stream] = layer.outputs[stream].clip(self.lower, self.upper)

class AddGaussianNoise(Decorator_ABC):
    """Add gaussian noise to the output of the layer"""
    
    def __init__(self, std, avg=1, strems=["train"], **kwargs):
        assert std > 0 :
        super(AddGaussianNoise, self).__init__(streams, **kwargs) 
        self.setHP("std", std)
        self.setHP("avg", avg)
        
    def run(self, layer, stream) :
        if stream in self.streams :
            rnd = tt.shared_randomstreams.RandomStreams()
            randomVals = rnd.normal(size = outputs.shape, avg=self.avg, std=self.std)
            layer.outputs[stream] = layer.outputs[stream] + randomVals

class MultGaussianNoise(Decorator_ABC):
    """Multiply gaussian noise to the output of the layer"""
    
    def __init__(self, std, avg=1, strems=["train"], **kwargs):
        assert std > 0 :
        super(MultGaussianNoise, self).__init__(streams, **kwargs) 
        self.setHP("std", std)
        self.setHP("avg", avg)
        
    def run(self, layer, stream) :
        if stream in self.streams :
            rnd = tt.shared_randomstreams.RandomStreams()
            randomVals = rnd.normal(size = outputs.shape, avg=self.avg, std=self.std)
            layer.outputs[stream] = layer.outputs[stream] * randomVals