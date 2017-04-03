import theano.tensor as tt
from Mariana.abstraction import Abstraction_ABC

__all__ = ["Activation_ABC", "Pass", "Sigmoid", "Tanh", "ReLU", "Softmax"]

class Activation_ABC(Abstraction_ABC):
    """All activations must inherit from this class"""

    def apply(self, layer, x) :
        """Apply to a layer and update networks's log"""
        
        message = "%s uses activation %s" % (layer.name, self.__class__.__name__)
        layer.network.logLayerEvent(layer, message, self.getHyperParameters())
        for s in x.streams :
            x[s] = self.run(x[s])

    def run(self, x) :
        """the actual activation run that will be applied to the neurones."""
        raise NotImplemented("Must be implemented in child")

class Pass(Activation_ABC):
    """
    simply returns x
    """
    def __init__(self):
        Activation_ABC.__init__(self)
        
    def run(self, x):
        return x

class Sigmoid(Activation_ABC):
    """
    .. math::

        1/ (1/ + exp(-x))"""
    def __init__(self):
        Activation_ABC.__init__(self)
        
    def run(self, x):
        return tt.nnet.sigmoid(x)

class Tanh(Activation_ABC):
    """
    .. math::

        tanh(x)"""
    def __init__(self):
        Activation_ABC.__init__(self)

    def run(self, x):
        return tt.tanh(x)

class ReLU(Activation_ABC):
    """
    .. math::

        max(0, x)"""
    def __init__(self, leakiness=0):
        Activation_ABC.__init__(self)
        self.setHP("leakiness", leakiness)

    def run(self, x):
        tt.nnet.relu(x, alpha=self.getHP("temperature"))
        # return tt.maximum(0., x)

class Softmax(Activation_ABC):
    """Softmax to get a probabilistic output
    
    .. math::

        scale * exp(x_i/T)/ sum_k( exp(x_k/T) )
    """
    def __init__(self, scale = 1, temperature = 1):
        Activation_ABC.__init__(self)
        self.setHP("temperature")
        self.setHP("scale")

    def run(self, x):
        return self.getHP("scale") * tt.nnet.softmax(x/self.getHP("temperature"))