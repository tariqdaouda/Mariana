import Mariana.initializations as MI

import Mariana.compatibility.lasagne as MLASAGNE
import lasagne.layers.pool as LasagnePOOL

__all__ = ["MaxPooling1D", "MaxPooling2D", "MaxPooling3D", "AveragePooling1D", "AveragePooling2D", "AveragePooling3D", "RepeatedUpscaling1D", "DilatedUpscaling1D", "RepeatedUpscaling2D", "DilatedUpscaling2D", "RepeatedUpscaling3D", "DilatedUpscaling3D", "WinnerTakesAll", "MaxSpatialPyramidPooling", "AverageSpatialPyramidPooling"]

class MaxPooling1D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's MaxPool1DLayer layer and performs a 1D max pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolSize,
        name,
        stride=None,
        pad=0,
        **kwargs
    ):
      
        super(MaxPooling1D, self).__init__(
                LasagnePOOL.MaxPool1DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": poolSize,
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                },
                name=name,

        )     

class MaxPooling2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's MaxPool2DLayer layer and performs a 2D max pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolHeight,
        poolWidth,
        name,
        stride=None,
        pad=(0, 0),
        **kwargs
    ):
        super(MaxPooling2D, self).__init__(
                LasagnePOOL.MaxPool2DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": (poolHeight, poolWidth),
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                },
                name=name,

        )       

class MaxPooling3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's MaxPool3DLayer layer and performs a 3D max pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolHeight,
        poolWidth,
        poolDepth,
        name,
        stride=None,
        pad=(0, 0, 0),
        **kwargs
    ):
        super(MaxPooling3D, self).__init__(
                LasagnePOOL.MaxPool3DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": (poolHeight, poolWidth, poolDepth),
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                },
                name=name,

        )

class AveragePooling1D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Pool1DLayer layer and performs a 1D average pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolSize,
        name,
        stride=None,
        pad=0,
        includePadding=False,
        **kwargs
    ):
        if includePadding :
            mode = "average_inc_pad"
        else:
            mode = "average_exc_pad"

        super(AveragePooling1D, self).__init__(
                LasagnePOOL.Pool1DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": poolSize,
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                    "mode": mode
                },
                name=name,

        )     

class AveragePooling2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Pool2DLayer layer and performs a 2D average pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolHeight,
        poolWidth,
        name,
        stride=None,
        pad=(0, 0),
        includePadding=False,
        **kwargs
    ):
        if includePadding :
            mode = "average_inc_pad"
        else:
            mode = "average_exc_pad"
        
        super(AveragePooling2D, self).__init__(
                LasagnePOOL.Pool2DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": (poolHeight, poolWidth),
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                    "mode": mode
                },
                name=name,
        )       

class AveragePooling3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Pool3DLayer layer and performs a 3D average pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolHeight,
        poolWidth,
        poolDepth,
        name,
        stride=None,
        pad=(0, 0, 0),
        includePadding=False,
        **kwargs
    ):
        if includePadding :
            mode = "average_inc_pad"
        else:
            mode = "average_exc_pad"

        super(AveragePooling3D, self).__init__(
                LasagnePOOL.Pool3DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": (poolHeight, poolWidth, poolDepth),
                    "stride": stride,
                    "pad": pad,
                    "ignore_border": True,
                    "mode": mode
                },
                name=name,
        )

class RepeatedUpscaling1D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale1DLayer layer and performs a 1D repeated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        scaleFactor,
        name,
        **kwargs
    ):

        super(RepeatedUpscaling1D, self).__init__(
                LasagnePOOL.Upscale1DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": scaleFactor,
                    "mode": "repeat"
                },
                name=name,
                
        )

class DilatedUpscaling1D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale1DLayer layer and performs a 1D dilated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        scaleFactor,
        name,
        **kwargs
    ):

        super(DilatedUpscaling1D, self).__init__(
                LasagnePOOL.Upscale1DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": scaleFactor,
                    "mode": "dilate"
                },
                name=name,
                
        )

class RepeatedUpscaling2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale2DLayer layer and performs a 2D repeated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        heightScaleFactor,
        widthScaleFactor,
        name,
        **kwargs
    ):

        super(RepeatedUpscaling2D, self).__init__(
                LasagnePOOL.Upscale2DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": (heightScaleFactor, widthScaleFactor),
                    "mode": "repeat"
                },
                name=name,
                
        )

class DilatedUpscaling2D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale2DLayer layer and performs a 2D dilated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        heightScaleFactor,
        widthScaleFactor,
        name,
        **kwargs
    ):

        super(DilatedUpscaling2D, self).__init__(
                LasagnePOOL.Upscale2DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": (heightScaleFactor, widthScaleFactor),
                    "mode": "dilate"
                },
                name=name,
                
        )

class RepeatedUpscaling3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale3DLayer layer and performs a 3D repeated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        heightScaleFactor,
        widthScaleFactor,
        depthScaleFactor,
        name,
        **kwargs
    ):

        super(RepeatedUpscaling3D, self).__init__(
                LasagnePOOL.Upscale3DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": (heightScaleFactor, widthScaleFactor, depthScaleFactor),
                    "mode": "repeat"
                },
                name=name,
                
        )

class DilatedUpscaling3D(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's Upscale3DLayer layer and performs a 3D dilated upscaling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        heightScaleFactor,
        widthScaleFactor,
        depthScaleFactor,
        name,
        **kwargs
    ):
        super(DilatedUpscaling3D, self).__init__(
                LasagnePOOL.Upscale3DLayer,
                initializations=[],
                lasagneHyperParameters={
                    "scale_factor": (heightScaleFactor, widthScaleFactor, depthScaleFactor),
                    "mode": "dilate"
                },
                name=name,
                
        )

class WinnerTakesAll(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's FeatureWTALayer layer and performs a Winner takes all pooling over each channel.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        poolSize,
        depthScaleFactor,
        name,
        axis=1,
        **kwargs
    ):

        super(WinnerTakesAll, self).__init__(
                LasagnePOOL.FeatureWTALayer,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": poolSize,
                    "axis": axis
                },
                name=name,
                
        )

class MaxSpatialPyramidPooling(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's SpatialPyramidPooling layer and performs a pyramid max pooling over each channel.
    This variant of max pooling can be applied on inputs of arbitrary lengths.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        poolDims=[4, 2, 1],
        implementation="fast",
        **kwargs
    ):

        super(MaxSpatialPyramidPooling, self).__init__(
                LasagnePOOL.SpatialPyramidPooling,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": poolSize,
                    "implementation": implementation,
                    "mode": 'max'
                },
                name=name,
                
        )


class AverageSpatialPyramidPooling(MLASAGNE.LasagneLayer):
    """This layer wraps lasagnes's SpatialPyramidPooling layer and performs a pyramid average pooling over each channel.
    This variant of average pooling can be applied on inputs of arbitrary lengths.
    For a full explanation of the arguments please checkout lasagne's doc"""
    def __init__(
        self,
        name,
        poolDims=[4, 2, 1],
        implementation="fast",
        includePadding=False,
        **kwargs
    ):
        if includePadding :
            mode = "average_inc_pad"
        else:
            mode = "average_exc_pad"

        super(AverageSpatialPyramidPooling, self).__init__(
                LasagnePOOL.SpatialPyramidPooling,
                initializations=[],
                lasagneHyperParameters={
                    "pool_size": poolSize,
                    "implementation": implementation,
                    "mode": mode
                },
                name=name,
                
        )

