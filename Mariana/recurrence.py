import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers as LasagneLayers

__all__ = ["RecurrentDense", "RecurrentLayer", "LSTM", "GatedRecurrentUnit", "GRU"]

class RecurrentDense(MLASAGNE.LasagneLayer):
    """The classical recurrent layer with dense input to hidden and hidden to hidden connections.
    For a full explanation of the arguments please checkout lasagne's doc on lasagne.layers.RecurrentLayer.
    """
    def __init__(
        self,
        size,
        name,
        initializations=[MI.Uniform('W_in_to_hid'), MI.Uniform('W_hid_to_hid'), MI.SingleValue('b', 0)],
        backwards=False,
        learnInit=False,
        gradientSteps=-1,
        gradClipping=0,
        unrollScan=False,
        # precomputeInput=False,
        onlyReturnFinal=False,
        **kwargs
    ):
        super(RecurrentDense, self).__init__(
                LasagneLayers.RecurrentLayer,
                lasagneHyperParameters={
                    "num_units": size,
                    "backwards": backwards,
                    "learn_init": learnInit,
                    "gradient_steps": gradientSteps,
                    "grad_clipping": gradClipping,
                    "unroll_scan": unrollScan,
                    "precompute_input": False,
                    "mask_input": None,
                    "only_return_final": onlyReturnFinal,
                },
                initializations=initializations,
                name=name,
                lasagneKwargs={},
                **kwargs
            )

        self.addHyperParameters(
            {
                # "maxSequenceLength": maxSequenceLength,
                "backwards": backwards,
                "learnInit": learnInit,
                "gradientSteps": gradientSteps,
                "gradClipping": gradClipping,
                "unrollScan": unrollScan,
                # "precomputeInput": precomputeInput,
                "onlyReturnFinal": onlyReturnFinal
            }
        )

    def getShape_abs(self) :
        if not self.inLayer :
            return None
        return self.lasagneLayer[self.streams[0]].get_output_shape_for([self.inLayer.getShape_abs()])

RecurrentLayer = RecurrentDense
