class Variable(object):
    """docstring for Variable"""
    def __init__(self, variable_type = None, streams=["train", "test"], *theano_args, **theano_kwargs):
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