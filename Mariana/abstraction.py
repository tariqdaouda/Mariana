import theano

__all__ = ["Abstraction_ABC"]

class Abstraction_ABC(object):
    """
    This class represents a layer modifier. This class must includes a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, *args, **kwargs):
        self.hyperParameters = []
        self.parameters = {}

    def apply(self, layer, cost) :
        """Apply to a layer and update networks's log"""
        raise NotImplemented("Must be implemented in child")
        
    def getParameterDict(self) :
        """returns the layer's parameters as dictionary"""
        from theano.compile import SharedVariable
        res={}
        for k, v in self.parameters.iteritems() :
            if isinstance(v, SharedVariable) :
                res[k]=v
        return res

    def getParameters(self) :
        """returns the layer's parameters"""
        return self.getParameterDict().values()

    def getParameterNames(self) :
        """returns the layer's parameters names"""
        return self.getParameterDict().keys()

    def getParameterShape(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplemented("Should be implemented in child")

    def toJson(self) :
        """A json representation of the object"""

        res = {
            "class": self.name,
            "hyperParameters": {}
        }
        for h in self.hyperParameters :
            res["hyperParameters"][h] = getattr(self, h)
        
        return res

    def __repr__(self) :
        return "< %s >" % self.__class__.__name__