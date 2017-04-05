import Mariana.useful as MUSE
import theano

class Variable(object):
    """docstring for Variable"""
    def __init__(self, streams=["train", "test"], variable_type = None, *theano_args, **theano_kwargs):
        super(Variable, self).__init__()
        self.streams = streams
        self.variables = {}
        for f in streams :
            if variable_type is not None:
                self.variables[f] = variable_type(*theano_args, **theano_kwargs)
            else :
                self.variables[f] = None

    def __getitem__(self, flow) :
        try :
            return self.variables[flow]
        except KeyError :
            raise KeyError("There is no flow by the name of: '%s'" % f)
    
    def __setitem__(self, flow, newVal) :
        try :
            self.variables[flow] = newVal
        except KeyError :
            raise KeyError("There is no flow by the name of: '%s'" % f)

class Inputs(Variable):
    """docstring for Input"""

class Targets(Variable):
    """docstring for Input"""

class Parameter(object):
    """docstring for Parameter"""
    def __init__(self, name, tags={"regularizable": True}):
        super(Parameter, self).__init__()
        self.name = name
        self.tags = tags
        self.value = None

    def hasValue(self) :
        return self.value is not None
    
    def setValue(self, v, forceReset=False) :
        if self.value is None or forceReset :
            self.value = theano.shared(value = MUSE.iCast_numpy(v), name = self.name)
        else :
            if v.shape != self.getShape() :
                print Warning("Update has a different shape: %s -> %s" %(self.shape, v.shape))
            self.value.set_value(MUSE.iCast_numpy(v))

    def getValue(self, v) :
        return self.get_value()

    def getShape(self) :
        return self.getValue().shape

    def hasTag(self, tag) :
        return tag in self.tags

    def getTag(self, tag) :
        return self.tags[tag]

    def setTag(self, tag, value = True) :
        self.tags[tag]

    def removeTag(self, tag) :
        del self.tags[tag]

    def __repr__(self) :
        return "< Parameter: %s, %s>" % (self.name, self.shape)

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
            self.store[k] = self.cost.apply(self.layer, self.targets[k], self.outputs[k], k)

    def __getitem__(self, k) :
        return self.store[k]

    def __setitem__(self, k, v) :
        self.store[k] = v