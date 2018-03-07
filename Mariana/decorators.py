import numpy
import theano
import theano.tensor as tt
import Mariana.settings as MSET
import Mariana.abstraction as MABS
import Mariana.initializations as MI
import Mariana.useful as MUSE
import Mariana.custom_types as MTYPES

__all__= ["Decorator_ABC", "BatchNormalization", "Center", "Normalize", "Mask", "RandomMask", "BinomialDropout", "Clip", "AdditiveGaussianNoise", "MultiplicativeGaussianNoise"]

class Decorator_ABC(MABS.TrainableAbstraction_ABC, MABS.Apply_ABC) :
    """A decorator is a modifier that is applied on a layer's output. They are always the last the abstraction to be applied."""

    def __init__(self, streams, **kwargs):
        super(Decorator_ABC, self).__init__(**kwargs)
        self.streams = set(self.streams)
        self.setHP("streams", streams)

    def apply(self, layer, stream) :
        """Apply to a layer and update networks's log"""
        if stream in self.streams :
	        return self.run(layer, stream=stream)

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
        rnd = tt.shared_randomstreams.RandomStreams()
        maskId = rnd.random_integers(low=0, high=self.nbMasks-1, ndim=1)
        mask = self.masks[maskId]

        layer.outputs[stream] = layer.outputs[stream] * mask

class BinomialDropout(Decorator_ABC):
    """Stochastically mask some parts of the output of the layer. Use it to make things such as denoising autoencoders and dropout layers"""
    def __init__(self, dropoutRatio, streams=["train"], **kwargs):
        super(BinomialDropout, self).__init__(streams, **kwargs)
        assert (dropoutRatio >= 0 and dropoutRatio <= 1)
        self.dropoutRatio = dropoutRatio
        self.seed = MSET.RANDOM_SEED
        self.setHP("dropoutRatio", dropoutRatio)
        
    def run(self, layer, stream) :        
        rnd = tt.shared_randomstreams.RandomStreams()
        mask = rnd.binomial(n = 1, p = (1-self.dropoutRatio), size = layer.outputs[stream].shape)
        # cast to stay in GPU float limit
        mask = MUSE.iCast_theano(mask)
        layer.outputs[stream] = layer.outputs[stream] * mask

class Center(Decorator_ABC) :
    """Centers the outputs by substracting the mean"""
    def __init__(self, streams=["train"], **kwargs):
        super(Center, self).__init__(streams, **kwargs)
        
    def run(self, layer, stream) :
        layer.outputs[stream] = layer.outputs[stream]-tt.mean(layer.outputs[stream])

class Normalize(Decorator_ABC) :
    """
    Normalizes the outputs by substracting the mean and dividing by the standard deviation

    :param float epsilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
    """

    def __init__(self, epsilon=1e-6, streams=["train"]) :
        super(Normalize, self).__init__(streams, **kwargs) 
        self.setHP("epsilon", epsilon)

    def run(self, layer, stream) :
        std = tt.sqrt( tt.var(layer.outputs[stream]) + self.epsilon )
        layer.outputs[stream] = ( layer.outputs[stream]-tt.mean(layer.output) / std )

# class BatchNormalization(Decorator_ABC):
#     """Applies Batch Normalization to the outputs of the layer.
#     Implementation according to Sergey Ioffe and Christian Szegedy (http://arxiv.org/abs/1502.03167)
    
#         .. math::

#            \\gamma * \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta

#         Where \\gamma and \\beta are learned and std stands for the standard deviation. The mean and the std are computed accross the whole minibatch.

#         :param float epsilon: Actually it is not the std that is used but the approximation: sqrt(Variance + epsilon). Use this parameter to set the epsilon value
#     """

#     def __init__(self, testMu, testSigma, initializations=[MI.SingleValue('gamma', 1), MI.SingleValue('beta', 0)], epsilon=1e-6, streams=["train", "test"], **kwargs) :
#         super(BatchNormalization, self).__init__(initializations=initializations, streams=streams, **kwargs)
#         self.setHP("testMu", testMu)
#         self.setHP("testSigma", testSigma)
#         self.setHP("epsilon", epsilon)
        
#         self.addParameters({
#             "gamma": MTYPES.Parameter("gamma"),
#             "beta": MTYPES.Parameter("beta")
#         })

#     def getParameterShape_abs(self, param, **kwargs) :
#         return self.parent.getShape_abs()

#     def run(self, layer, stream) :
#         if stream == "train" :
#             mu = tt.mean(layer.outputs[stream])
#             sigma = tt.sqrt( tt.var(layer.outputs[stream]) + self.getHP("epsilon") )
#         elif stream == "test" :
#             mu = self.getHP("testMu")
#             sigma = self.getHP("testSigma")
        
#         layer.outputs[stream] = self.getP("gamma")() * ( (layer.outputs[stream] - mu) / sigma ) + self.getP("beta")()

class Clip(Decorator_ABC):
    """Clips the neurone activations, preventing them to go beyond the specified range"""
    
    def __init__(self, lower, upper, streams=["train"], **kwargs) :
        super(Clip, self).__init__(streams, **kwargs) 
        assert lower < upper
        self.setHP("lower", lower)
        self.setHP("upper", upper)
    
    def run(self, layer, stream) :
        layer.outputs[stream] = layer.outputs[stream].clip(self.lower, self.upper)

class AddGaussianNoise(Decorator_ABC):
    """Add gaussian noise to the output of the layer"""
    
    def __init__(self, std, avg=0, streams=["train"], **kwargs):
        assert std > 0
        super(AddGaussianNoise, self).__init__(streams, **kwargs) 
        self.setHP("std", std)
        self.setHP("avg", avg)
        
    def run(self, layer, stream) :
        rnd = tt.shared_randomstreams.RandomStreams()
        randomVals = rnd.normal(size = layer.getIntrinsicShape(), avg=self.getHP("avg"), std=self.getHP("std") )
        layer.outputs[stream] = layer.outputs[stream] + randomVals

class MultGaussianNoise(Decorator_ABC):
    """Multiply gaussian noise to the output of the layer"""
    
    def __init__(self, std, avg=0, streams=["train"], **kwargs):
        assert std > 0
        super(MultGaussianNoise, self).__init__(streams, **kwargs) 
        self.setHP("std", std)
        self.setHP("avg", avg)
        
    def run(self, layer, stream) :
        rnd = tt.shared_randomstreams.RandomStreams()
        randomVals = rnd.normal(size = layer.getIntrinsicShape(), avg=self.getHP("avg"), std=self.getHP("std") )
        layer.outputs[stream] = layer.outputs[stream] * randomVals

class Scale(Decorator_ABC):
    """Multiplies the output by scale"""
    
    def __init__(self, scale, streams=["train", "test"], **kwargs):
        super(Scale, self).__init__(streams, **kwargs) 
        self.setHP("scale", scale)
        
    def run(self, layer, stream) :
        layer.outputs[stream] = layer.outputs[stream] * self.getHP("scale")

class Shift(Decorator_ABC):
    """Shifts (addiction) the output by scale"""
    
    def __init__(self, shift, streams=["train", "test"], **kwargs):
        super(Shift, self).__init__(streams, **kwargs) 
        self.setHP("shift", shift)
        
    def run(self, layer, stream) :
        layer.outputs[stream] = layer.outputs[stream] + self.getHP("shift")