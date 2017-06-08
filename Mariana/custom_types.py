import Mariana.useful as MUSE
import theano

class Variable(object):
    """docstring for Variable"""
    def __init__(self, variableType=None, streams=["train", "test"], *theano_args, **theano_kwargs):
        super(Variable, self).__init__()
        self.streams = streams
        self.variables = {}
        self.variableType = variableType
        if variableType :
            self.set(variableType, *theano_args, **theano_kwargs)
        else :
            self.dtype = None
            for f in self.streams :
                self.variables[f] = None    
    
    def set(self, variableType, *theano_args, **theano_kwargs) :
        self.variableType = variableType
        for f in self.streams :
            self.variables[f] = variableType(*theano_args, **theano_kwargs)
        self.dtype = self.variables[f].dtype
    
    def getValue(self, stream) :
        v = self[stream]
        if v is None :
            raise ValueError("Variable has an empty value for stream: %s" % stream)

        return self[stream].get_value()

    def setValue(self, stream, value) :
        self[stream].set_value(value)

    def __getitem__(self, stream) :
        try :
            return self.variables[stream]
        except KeyError :
            raise KeyError("There is no stream by the name of: '%s'" % f)
    
    def __setitem__(self, stream, newVal) :
        try :
            self.variables[stream] = newVal
        except KeyError :
            raise KeyError("There is no stream by the name of: '%s'" % f)

class Inputs(Variable):
    """docstring for Input"""

class Targets(Variable):
    """docstring for Input"""

class Parameter(object):
    """docstring for Parameter"""
    def __init__(self, name):
        super(Parameter, self).__init__()
        self.name = name
        self.value = None

    def __call__(self) :
        return self.value

    def hasValue(self) :
        return self.value is not None
    
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

        self.value = v

    def updateValue(self, value, forceCast=False) :
        if forceCast :
            v = theano.shared(value = MUSE.iCast_numpy(value), name = self.name)
        else :
            v = value

        if v.shape != self.getShape() :
            print("Warning update has a different shape: %s -> %s" %(self.shape, v.shape))
        self.value.set_value(v)
    
    def getValue(self) :
        if self.value is None :
            return None
        return self.value.get_value()

    def getShape(self) :
        if self.value is None :
            return None
        return self.getValue().shape

    def __repr__(self) :
        return "< Parameter: %s, %s>" % (self.name, self.getShape())

class Losses(object):
    """docstring for Losses"""
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