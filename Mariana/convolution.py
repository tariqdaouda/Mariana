import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers.conv as LasagneCONV

__all__ = ["Convolution1D", "Convolution2D", "Convolution3D", "TransposeConvolution2D", "Deconv2D", "TransposeConvolution3D", "Deconv3D", "DilatedConv2DLayer"]

class Convolution1D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Conv1DLayer layer and performs a 1D convolution over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterSize,
        stride=1,
        pad=0,
        untieBiases=False,
        flipFilters=True,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        **kwargs
    ):
        super(Convolution1D, self).__init__(
                LasagneCONV.Conv1DLayer,
                lasagneHyperParameters={
                    "num_filters": numFilters,
                    "filter_size": filterSize,
                    "stride": stride,
                    "pad": pad,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                **kwargs
            )
    
class Convolution2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Conv2DLayer layer and performs a 2D convolution over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterHeight,
        filterWidth,
        stride=(1, 1),
        pad=0,
        untieBiases=False,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        flipFilters=True,
        **kwargs
    ):
        super(Convolution2D, self).__init__(
                LasagneCONV.Conv2DLayer,
                lasagneHyperParameters={
                    "num_filters": numFilters,
                    "filter_size": (filterHeight, filterWidth),
                    "stride": stride,
                    "pad": pad,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                name=name,
                **kwargs
            )

class Convolution3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Conv3DLayer layer and performs a 3D convolution over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterHeight,
        filterWidth,
        filterDepth,
        stride=(1, 1, 1),
        pad=0 ,
        untieBiases=False,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        flipFilters=True,
        **kwargs
    ):
        super(Convolution3D, self).__init__(
                LasagneCONV.Conv3DLayer,
                lasagneHyperParameters={
                    "numFilters": numFilters,
                    "filter_size": (filterHeight, filterWidth, filterDepth),
                    "stride": stride,
                    "pad": pad,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                name=name,
                **kwargs
            )

class TransposeConvolution2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's TransposedConv2DLayer layer and performs a 2D transpose convolution (deconvolution) over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterHeight,
        filterWidth,
        stride=(1, 1),
        crop=0,
        untieBiases=False,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        flipFilters=True,
        **kwargs
    ):
        super(TransposeConvolution2D, self).__init__(
                LasagneCONV.TransposedConv2DLayer,
                lasagneHyperParameters={
                    "num_filters": numFilters,
                    "filter_size": (filterHeight, filterWidth),
                    "stride": stride,
                    "crop": crop,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                name=name,
                **kwargs
            )
Deconv2D = TransposeConvolution2D

class TransposeConvolution3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's TransposedConv3DLayer layer and performs a 3D transpose convolution (deconvolution) over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterHeight,
        filterWidth,
        filterDepth,
        stride=(1, 1, 1),
        crop=0 ,
        untieBiases=False,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        flipFilters=True,
        **kwargs
    ):
        super(Convolution3D, self).__init__(
                LasagneCONV.TransposedConv3DLayer,
                lasagneHyperParameters={
                    "num_filters": numFilters,
                    "filter_size": (filterHeight, filterWidth, filterDepth),
                    "stride": stride,
                    "crop": crop,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                name=name,
                **kwargs
            )
Deconv3D = TransposeConvolution3D
    
class DilatedConvolution2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's DilatedConv2DLayer layer and performs a 2D dilated convolution (deconvolution) over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        numFilters,
        filterHeight,
        filterWidth,
        dilation=(1, 1),
        stride=(1, 1),
        pad=0,
        untieBiases=False,
        initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
        flipFilters=True,
        **kwargs
    ):
        super(TransposeConvolution2D, self).__init__(
                LasagneCONV.DilatedConv2DLayer,
                lasagneHyperParameters={
                    "num_filters": numFilters,
                    "filter_size": (filterHeight, filterWidth),
                    "stride": stride,
                    "pad": pad,
                    "dilation": dilation,
                    "untie_biases": untieBiases,
                    "flip_filters": flipFilters
                },
                lasagneKwargs={},
                name=name,
                **kwargs
            )