import Mariana.abstraction as MABS

__all__ = ["SingleLayerRegularizer_ABC", "L1", "L2", "ActivationL1"]
        
class SingleLayerRegularizer_ABC(MABS.UntrainableAbstraction_ABC, MABS.Apply_ABC) :
    """An abstract regularization to be applied to a layer."""

    def __init__(self, streams=["train"], **kwargs):
        self.streams = streams
        super(SingleLayerRegularizer_ABC, self).__init__(**kwargs)
        
    def apply(self, layer, variable, stream) :
        """Apply to a layer and update networks's log"""

        if stream in self.streams :
            variable += self.run(layer, stream)

        return variable
        
    def run(self, layer, stream) :
        """Returns the expression to be added to the cost"""
        raise NotImplemented("Must be implemented in child")

class L1(SingleLayerRegularizer_ABC) :
    """
    Will add this to the cost. Weights will tend towards 0
    resulting in sparser weight matrices.
    .. math::

            factor * abs(Weights)
    """
    def __init__(self, factor) :
        super(L1, self).__init__()
        self.setHP("factor", factor)

    def run(self, layer, stream) :
        return self.getHP("factor") * ( abs(layer.getP("W")()).sum() )

class L2(SingleLayerRegularizer_ABC) :
    """
    Will add this to the cost. Causes the weights to stay small
    .. math::

            factor * (Weights)^2
    """
    def __init__(self, factor) :
        super(L2, self).__init__()
        self.setHP("factor", factor)

    def run(self, layer, stream) :
        return self.getHP("factor") * ( (layer.getP("W")() ** 2).sum() )

class ActivationL1(SingleLayerRegularizer_ABC) :
    """
    L1 on the activations. Neurone activations will tend towards
    0, resulting into sparser representations.

    Will add this to the cost
    .. math::

            factor * abs(activations)
    """
    def __init__(self, factor, stream) :
        super(ActivationL1, self).__init__()
        SingleLayerRegularizer_ABC.__init__(self)
        self.setHP("factor", factor)

    def run(self, layer) :
        return self.getHP("factor") * ( abs(layer.outputs[streams]).sum() )