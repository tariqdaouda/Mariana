from collections import OrderedDict
import Mariana.custom_types as MTYPES

__all__ = ["Abstraction_ABC", "ApplyAbstraction_ABC"]

class Logger_ABC(object):
    """Interface for objects that log events"""
    def __init__(self, **kwargs):
        super(Logger_ABC, self).__init__()
        self.log = []

    def logEvent(self, message, **moreStuff) :
        import time
            
        entry = {
            "date": time.ctime(),
            "timestamp": time.time(),
            "message": message,
        }
        entry.update(moreStuff)
        self.log.append(entry)

    def getLog(self) :
        raise NotImplemented("Must be implemented in child")

    def printLog(self) :
        """JSON pretty printed log"""
        import json
        print json.dumps(self.getLog(), indent=4, sort_keys=True)

class Abstraction_ABC(Logger_ABC):
    """
    This class represents a layer modifier. This class must includes a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, streams=["train", "test"], **kwargs):
        super(Abstraction_ABC, self).__init__()
        self.name = self.__class__.__name__
        
        self.streams = streams
        self.hyperParameters = OrderedDict()
        self.parameters = OrderedDict()
        self.notes = OrderedDict()

        self.abstractions={
            "initializations": [],
        }

        self._mustInit=True

    def getLog(self) :
        return self.log

    def addNote(self, title, text) :
        self.notes[title] = text

    def addHyperParameters(self, dct) :
        """adds to the list of hyper params, dct must be a dict"""
        self.hyperParameters.update(dct)
    
    def addParameters(self, dct) :
        for k, v in dct.iteritems() :
            self.setP(k, v)

    def setHyperparameter(k, v) :
        """sets a single hyper parameter"""
        self.setHP(k, v)

    def getHyperparameter(k, v) :
        """get a single hyper parameter"""
        self.getHP(k, v)

    def setHP(self, k, v) :
        """setHyperparameter() alias"""
        self.hyperParameters[k] = v

    def getHP(self, k) :
        """getHyperparameter() alias"""
        return self.hyperParameters[k]

    def setParameter(k, v) :
        """Brutally set the value of a parameter. No checks applied"""
        self.setP(k, v)

    def getParameter(k, v) :
        """get a single parameter"""
        self.getP(k, v)

    def setP(self, param, value) :
        """setParameter() alias"""
        if isinstance(value, MTYPES.Parameter) :
            self.parameters[param] = value
        else :
            self.parameters[param].setValue(value)
    
    def getP(self, k) :
        """getParameter() alias"""
        return self.parameters[k]

    def getParameterShape_abs(self, param) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplemented("Should be implemented in child")

    def _initParameters(self, forceReset=False) :
        """creates the parameters if necessary"""
        if self._mustInit or forceReset :
            for init in self.abstractions["initializations"] :
                init._apply(self)
        self._mustInit=False

    # def __getattribute__(self, k) :
    #     if k == "__getstate__" or k == "__slots__" :
    #         print "----", k, self
    #         print ">>>", super(Abstraction_ABC, self).__getattribute__(k) is None
    #         print "========="

    #     hps = object.__getattribute__(self, "hyperParameters")
    #     if k in hps :
    #         return hps[k]
    #     ps = super(Abstraction_ABC, self).__getattribute__("parameters")
    #     if k in ps :
    #         return ps[k]
    #     # try :
    #     # ret = super(Abstraction_ABC, self).__getattribute__(k)
    #     # print "=====", k, ret
    #     # return ret
    #     # except AttributeError :
    #         # raise AttributeError("Abstraction of Class '%s' has no attribute '%s'" % (self.__class__.__name__, k))

    def toDictionary(self) :
        """A dct representation of the object"""
        if self.__class__ is Abstraction_ABC :
            raise AttributeError("This function cannot be launched from an instance of Abstraction_ABC")

        res = {
            "name": str(self.name),
            "hyperParameters": OrderedDict(self.hyperParameters),
            "notes": OrderedDict(self.notes),
        }
        ps = OrderedDict()
        for k, v in self.parameters.iteritems() :
            ps[k] = {"shape": self.getParameterShape_abs(k)}

        res["parameters"] = ps    
        
        return res

    def __repr__(self) :
        return "< %s: %s >" % (self.__class__.__name__, dict(self.hyperParameters))

class ApplyAbstraction_ABC(Abstraction_ABC):

    def __init__(self, **kwargs):
        super(ApplyAbstraction_ABC, self).__init__()
        self._mustInit=True

    def _initialize(self, layer) :
        if self._mustInit :
            self.initialize(layer)
            self._mustInit=False

    def initialize(self, layer) :
        """Last setup before apply, default: does nothing. Parameter initializations must be put here"""
        pass

    def logApply(self, layer, **kwargs) :
        message = "Applying : '%s' on layer '%s'" % (self.name, layer.name)
        self.logEvent(message)
        
    def _apply(self, layer, **kwargs) :
        """does self.set() + self.apply()"""
        self._initialize(layer)
        self.logApply(layer)
        self.apply(layer, **kwargs)

    def apply(self, layer, **kwargs) :
        """Apply to a layer, basically logs stuff and then calls run"""
        raise NotImplemented("Must be implemented in child")
    
    def run(self, **kwargs) :
        """the actual worker function that does whaters the abstraction is supposed to do"""
        raise NotImplemented("Must be implemented in child")
