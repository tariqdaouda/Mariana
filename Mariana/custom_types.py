class Variable(object):
    """docstring for Variable"""
    def __init__(self, variable_type = None, streams=["train", "test"]):
        super(Variable, self).__init__()
        self.variables = {}
        for f in streams :
            if variable_type is not None:
                self.variables[f] = variable_type()
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