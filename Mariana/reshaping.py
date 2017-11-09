import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers as LasagneLayers

__all__ = ["Reshape", "Flatten", "Dimshuffle", "Padding", "Slice"]

class Reshape(MLASAGNE.LasagneLayer):
    """
    reshape the output of a layer to a new shape.
    newShape can be a tuple or a TensorVariable.
    Each elements of the the tuple can be:
        * a positive int to denote the size of the dimension
        * a list of a single element, ex: [i],  to use the size of of the ith input element
        * -1 To infer the size of the dimension based on the sizes of all other dimension. There cannot be more than one -1.
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
        # self.shape = newShape