import theano.tensor as tt
import Mariana.abstraction as MABS

__all__ = ["Activation_ABC", "Pass", "Sigmoid", "Tanh", "ReLU", "Softmax"]

class Activation_ABC(MABS.TrainableAbstraction_ABC, MABS.Apply_ABC):
    """All activations must inherit from this class"""

    def apply(self, layer, x) :
        """Apply to a layer and update networks's log"""
        for s in x.streams :
            x[s] = self.run(x[s])

    def run(self, x) :
        """the actual activation run that will be applied to the neurones."""
        raise NotImplemented("Must be implemented in child")

class Pass(Activation_ABC):
    """
    simply returns x
    """
    
    def run(self, x):
        return x
Linear = Pass

class Sigmoid(Activation_ABC):
    """
    .. math::

        1/ (1/ + exp(-x))"""
    def run(self, x):
        return tt.nnet.sigmoid(x)

class Swish(Activation_ABC):
    """
    .. math::

        x * (1/ (1/ + exp(-x)) )"""
    def run(self, x):
        return x*tt.nnet.sigmoid(x)

class Softplus(Activation_ABC):
    """
    .. math::

        ln(1 + exp(x))"""
    def run(self, x):
        return tt.nnet.softplus(x)

class Sin(Activation_ABC):
    """
    .. math::

        x * (1/ (1/ + exp(-x)) )"""
    def run(self, x):
        return tt.Sin(x)

class Tanh(Activation_ABC):
    """
    .. math::

        tanh(x)"""
    def run(self, x):
        return tt.tanh(x)

class ReLU(Activation_ABC):
    """
    .. math::

        if pre_act < 0 return leakiness; else return pre_act"""
    def __init__(self, leakiness=0):
        super(ReLU, self).__init__()
        self.setHP("leakiness", leakiness)

    def run(self, x):
        return tt.nnet.relu(x, alpha=self.getHP("leakiness"))

class Softmax(Activation_ABC):
    """Softmax to get a probabilistic output
    
    .. math::

        scale * exp(x_i/T)/ sum_k( exp(x_k/T) )
    """
    def __init__(self, scale = 1, temperature = 1):
        super(Softmax, self).__init__()
        self.setHP("temperature", temperature)
        self.setHP("scale", scale)

    def run(self, x):
        return self.getHP("scale") * tt.nnet.softmax(x/self.getHP("temperature"))
