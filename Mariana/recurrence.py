import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers as LasagneLayers

__all__ = ["RecurrentDense", "LSTM", "GatedRecurrentUnit", "GRU"]

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

class GateConfig(object):
    """Legacy from lasagne. Holds the configuration for a gate but in Mariana way."""
    #     def __init__(self,
    #     W_in=MI.Normal('W_in', 0.1, 0),
    #     W_hid=MI.Normal('W_hid', 0.1, 0),
    #     W_in_initialization=MI.Normal('W_cell', 0.1, 0),
    #     W_in_initialization=MI.SingleValue('b', 0),
    #     activation=MI.Sigmoid()
    # ):
        
    def __init__(self,
        initializations=[MI.Normal('W_in', 0.1, 0), MI.Normal('W_hid', 0.1, 0), MI.Normal('W_cell', 0.1, 0), MI.SingleValue('b', 0)],
        activation=MI.Sigmoid()
    ):
        
        super(GateConfig, self).__init__()
        self.initializations = initializations
        self.activation = activation
        

class LSTM(LasagneLayers.LSTMLayer) :
    def __init__(
        self,
        size,
        name,
        inGateConfig=GateConfig(),
        forgateGateConfig=GateConfig(),
        cellGateConfig=GateConfig(),
        outgateGateConfig=GateConfig(W_cell=None, activation=MA.Tanh()),
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
    
# class lasagne.layers.LSTMLayer(incoming,
#     # num_units,
#     ingate=lasagne.layers.Gate(),
#     forgetgate=lasagne.layers.Gate(),
#     cell=lasagne.layers.Gate( W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
#     outgate=lasagne.layers.Gate(),
#     # nonlinearity=lasagne.nonlinearities.tanh,
#     cell_init=lasagne.init.Constant(0.),
#     hid_init=lasagne.init.Constant(0.),
#     peepholes=True,
#     # backwards=False,
#     # learn_init=False,
#     # gradient_steps=-1,
#     # grad_clipping=0,
#     # unroll_scan=False,
#     # precompute_input=True,
#     # mask_input=None,
#     # only_return_final=False,
#     **kwargs
# )