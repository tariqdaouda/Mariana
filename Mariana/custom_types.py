import theano
import Mariana.useful as MUSE

class Variable(object):
    """docstring for Variable"""
    def __init__(self, variableType=None, streams=["train", "test"], **theano_kwargs):
        super(Variable, self).__init__()
        self.streams = streams
        self.variables = {}
        self.variableType = variableType
        if variableType :
            self.set(variableType, **theano_kwargs)
        else :
            self.dtype = None
            for f in self.streams :
                self.variables[f] = None    
        
        self.tied = False

    def isTied(self) :
        return self.tied

    def tie(self, var) :
        if self.streams != var.streams :
            raise ValueError( "%s does not have the same streams. Self: %s, var: %s" % (var, self.streams, var.streams) )

        for f in self.streams :
            self.variables[f] = var.variables[f]
        self.tied = True
        
    def set(self, variableType, *theano_args, **theano_kwargs) :
        self.variableType = variableType
        for f in self.streams :
            self.variables[f] = variableType(*theano_args, **theano_kwargs)
        self.dtype = self.variables[f].dtype
        self.tied = False
    
    def getValue(self, stream) :
        v = self[stream]
        if v is None :
            raise ValueError("Variable has an empty value for stream: %s" % stream)

        return self[stream].get_value()

    def setValue(self, stream, value) :
        if stream not in self.streams :
            raise KeyError("There is no stream by the name of: '%s'" % stream)
        
        self[stream].set_value(value)
        self.tied = False

    def __getitem__(self, stream) :
        try :
            return self.variables[stream]
        except KeyError :
            raise KeyError("There is no stream by the name of: '%s'" % stream)
    
    def __setitem__(self, stream, newVal) :
        try :
            self.variables[stream] = newVal
        except KeyError :
            raise KeyError("There is no stream by the name of: '%s'" % stream)
        self.tied = False

    def __contains__(self, stream) :
        """check if the stream is supported"""
        return stream in self.streams

    def __repr__(self) :
        return "< Mariana %s, streams: %s>" % (self.__class__.__name__, self.streams)

class Inputs(Variable):
    """docstring for Input"""

class Targets(Variable):
    """docstring for Input"""

class Parameter(object):
    """docstring for Parameter"""
    def __init__(self, name):
        super(Parameter, self).__init__()
        self.name = name
        self.theano_var = None

    def __call__(self) :
        return self.getVar()

    def getVar(self) :
        return self.theano_var

    def hasValue(self) :
        return self.theano_var is not None
    
    def setValue(self, value, forceCast = True) :
        if isinstance(value, theano.Variable) :
            if forceCast :
                v = MUSE.iCast_theano(value)
            else :
                v = value
        else :
            if forceCast :
                v = theano.shared(value = MUSE.iCast_numpy(value), name = self.name)
            else :
                v = theano.shared(value = value, name = self.name)

        self.theano_var = v

    def updateValue(self, value, forceCast=False) :
        if forceCast :
            v = theano.shared(value = MUSE.iCast_numpy(value), name = self.name)
        else :
            v = value

        if v.shape != self.getShape() :
            print("Warning update has a different shape: %s -> %s" %(self.shape, v.shape))
        self.theano_var.set_value(v)
    
    def getValue(self) :
        if self.theano_var is None :
            return None
        return self.theano_var.get_value()

    def getShape(self) :
        if self.theano_var is None :
            return None
        return self.getValue().shape

    def __repr__(self) :
        return "< Mariana Parameter: %s, %s>" % (self.name, self.getShape())

class Losses(object):
    """Contains the loss for every stream"""
    def __init__(self, layer, cost, targets, outputs):
        super(Losses, self).__init__()
        self.streams=targets.streams

        self.layer = layer
        self.cost = cost
        self.targets = targets
        self.outputs = outputs

        self.store = {}
        for k in self.streams :
            self.store[k] = self.cost.apply(self.layer, self.targets[k], self.outputs[k], stream = k)

    def __getitem__(self, k) :
        return self.store[k]

    def __setitem__(self, k, v) :
        self.store[k] = v

    def __contains__(self, stream) :
        """check if the stream is supported"""
        return stream in self.streams

    def __repr__(self) :
        return "< Mariana %s, streams: %s>" % (self.__class__.__name__, self.streams)

