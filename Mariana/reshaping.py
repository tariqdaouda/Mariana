import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers as LasagneLayers

__all__ = ["Reshape", "Flatten", "Dimshuffle", "Padding", "Slice"]

class Reshape(MLASAGNE.LasagneLayer):
    """The classical recurrent layer with dense input to hidden and hidden to hidden connections.
    For a full explanation of the arguments please checkout lasagne's doc on lasagne.layers.RecurrentLayer.
    """
    def __init__(
        self,
        newShape,
        name=None,
        **kwargs
    ):
        super(Reshape, self).__init__(
                LasagneLayers.ReshapeLayer,
                lasagneHyperParameters={
                    "shape": newShape,
                },
                initializations=[],
                name=name,
                lasagneKwargs={},
                **kwargs
            )
