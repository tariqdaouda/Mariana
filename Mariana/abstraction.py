from collections import OrderedDict

__all__ = ["Abstraction_ABC", "ApplyAbstraction_ABC"]

        
class Abstraction_ABC(object):
    """
    This class represents a layer modifier. This class must includes a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, *args, **kwargs):
        super(Abstraction_ABC, self).__init__()
        self.hyperParameters = OrderedDict()
        self.parameters = OrderedDict()

    def addHyperParameters(self, dct) :
        """adds to the list of hyper params, dct must be a dict"""
        self.hyperParameters.update(dct)
    
    def addParameters(self, dct) :
        self.parameters.update(dct)

    def setHP(self, k, v) :
        """sets a single hyper parameter"""
        self.hyperParameters[k] = v

    def setP(self, k, v) :
        """sets a single parameter"""
        self.parameter[k] = v

    def getParameterShape(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplemented("Should be implemented in child")

    def __getattr__(self, k) :
        hps = object.__getattribute__(self, "hyperParameters")
        ps = object.__getattribute__(self, "parameters")
        if k in hps :
            return hps[k]
        if k in ps :
            return ps[k]
        raise AttributeError("Abstraction of Class '%s' has no attribute '%s'" % (self.__class__.__name__, k))

    # def toJson(self) :
    #     """A json representation of the object"""

    #     res = {
    #         "class": self.name,
    #         "hyperParameters": {}
    #     }
    #     for h in self.hyperParameters :
    #         res["hyperParameters"][h] = getattr(self, h)
        
    #     return res
    def __repr__(self) :
        return "< %s: %s >" % (self.__class__.__name__, self.hyperParameters)


class ApplyAbstraction_ABC(Abstraction_ABC):
    def apply(self, layer, *args, **kwargs) :
        """Apply to a layer, basically logs stuff and then calls run"""
        raise NotImplemented("Must be implemented in child")
    
    def run(self, *args, **kwargs) :
        """the actual worker function that does whaters the abstraction is supposed to do"""
        raise NotImplemented("Must be implemented in child")