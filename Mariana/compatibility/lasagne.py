from collections import OrderedDict
import Mariana.layers as ML
import Mariana.custom_types as MTYPES
import lasagne

__all__=["LasagneLayer", "LasagneStreamedLayer", "IAmAnnoyed"]

class IAmAnnoyed(Exception) :
    """What Mariana raises when you annoy her"""
    def __init__(self, msg) :
        self.message = msg

    def __str__(self) :
        return self.message
        
    def __repr__(self):
        return self.message

class LasagneStreamedLayer(object):
    """Wraps lasagne layers to give them a stream interface"""
    def __init__(self, incomingShape, streams, lasagneLayerCls, hyperParameters, initParameters, lasagneKwargs={}):
        super(LasagneStreamedLayer, self).__init__()
        self.streams = streams
        self.lasagneLayerCls = lasagneLayerCls
        self.hyperParameters = hyperParameters
        self.initParameters = initParameters
        self.lasagneKwargs = lasagneKwargs
        
        self.lasagneLayer = OrderedDict()

        kwargs = {}
        kwargs.update(self.hyperParameters)
        kwargs.update(self.initParameters)
        kwargs.update(self.lasagneKwargs)

        self.parameters = {}
        # print "lkjhfasd", incomingShape
        for f in streams :
            if len(self.parameters) == 0 :
                self.lasagneLayer[f] = self.lasagneLayerCls(incoming = incomingShape, **kwargs)
                for k in self.initParameters :
                    self.parameters[k] = getattr(self.lasagneLayer[f], k)
                kwargs.update(self.parameters)
            else :
                self.lasagneLayer[f] = self.lasagneLayerCls(incoming = incomingShape, **kwargs)

    def __getitem__(self, k) :
        return self.lasagneLayer[k]

    def __setitem__(self, k, v) :
        self.lasagneLayer[k] = v

class LasagneLayer(ML.Layer_ABC) :
    """This very special class allows you to incorporate a Lasagne layer seemlessly inside a Mariana network.
    An incorporated lasagne is just like a regular layer, with streams and all the other Mariana niceties.
    Initializations must be specified with Mariana initializers, and please don't pass it an 'incoming', 'nonlinearity' argument.
    It is Mariana's job to do the shape inference and activate the layers, and she can get pretty upset if you try to tell her how to do her job.
    If you need to specifiy a specific value for some paramters, use the HardSet() initializer.

    Here's an examples::

        from lasagne.layers.dense import DenseLayer

        hidden = LasagneLayer(
            DenseLayer,
            lasagneHyperParameters={"num_units": 10},
            initializations=[MI.GlorotNormal('W'), MI.SingleValue('b', 0)],
            activation = MA.Tanh(),
            learningScenari = [MS.GradientDescent(lr = 0.1, momentum=0)],
            name = "HiddenLayer2"
        )
    
    """

    def __init__(self, lasagneLayerCls, lasagneHyperParameters={}, lasagneKwargs={}, **kwargs) :
        import inspect

        super(LasagneLayer, self).__init__(**kwargs)

        self.lasagneLayerCls = lasagneLayerCls

        self.lasagneHyperParameters = lasagneHyperParameters
        
        if "nonlinearity" in self.lasagneHyperParameters :
            raise IAmAnnoyed("There's an 'nonlinearity' argument in the hyperParameters. Use activation = <...>. Just like you would do for any other layer.")
        
        if "incoming" in self.lasagneHyperParameters :
            raise IAmAnnoyed("There's an 'incoming' argument in the hyperParameters. Don't tell me how to do my job!")
        
        self.addHyperParameters(self.lasagneHyperParameters)
        if "nonlinearity" in inspect.getargspec(lasagneLayerCls.__init__)[0] :
            self.lasagneHyperParameters["nonlinearity"] = None
        self.lasagneKwargs = lasagneKwargs
        
        self.lasagneLayer  = None
        self.inLayer = None
        self.lasagneParameters = {}
        
        for init in self.abstractions["initializations"] :
            self.setP(init.getHP("parameter"), MTYPES.Parameter("%s.%s" % (self.name, init.getHP("parameter"))))
            init.setup(self)
            self.lasagneParameters[init.getHP("parameter")] = init.run

    def femaleConnect(self, layer) :
        self.inLayer = layer
        # print "---=-", self.inLayer.getShape_abs()
        # print self.lasagneParameters
        if not self.lasagneLayer :
            self.lasagneLayer = LasagneStreamedLayer(incomingShape=self.inLayer.getShape_abs(), streams=self.streams, lasagneLayerCls=self.lasagneLayerCls, hyperParameters=self.lasagneHyperParameters, initParameters=self.lasagneParameters)
        
    def getShape_abs(self) :
        if not self.inLayer :
            return None
        return self.lasagneLayer[self.streams[0]].get_output_shape_for(self.inLayer.getShape_abs())

    def _initParameters(self) :
        for k, v in self.lasagneLayer.parameters.iteritems() :
            self.parameters[k].setValue(v, forceCast=False)

    def getParameterShape_abs(self, k) :
        v = getattr(self.lasagneLayer[self.streams[0]], k)
        return v.get_value().shape
    
    def setOutputs_abs(self) :
        for f in self.outputs.streams :
            self.outputs[f] = self.lasagneLayer[f].get_output_for(self.inLayer.outputs[f])