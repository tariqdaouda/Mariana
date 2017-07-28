from collections import OrderedDict
import Mariana.custom_types as MTYPES

__all__ = ["Logger_ABC", "Abstraction_ABC", "UntrainableAbstraction_ABC", "TrainableAbstraction_ABC", "Apply_ABC"]

class Logger_ABC(object):
    """Interface for objects that log events"""
    def __init__(self, **kwargs):
        super(Logger_ABC, self).__init__()
        self.log = []
        self.notes = OrderedDict()

    def logEvent(self, message, **moreStuff) :
        """log an event"""
        import time
            
        entry = {
            "date": time.ctime(),
            "timestamp": time.time(),
            "message": message,
        }
        entry.update(moreStuff)
        self.log.append(entry)

    def getLog(self) :
        """return the log"""
        return self.log

    def addNote(self, title, text) :
        """add a note"""
        self.notes[title] = text

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
        
        self.streams = streams
        self.hyperParameters = OrderedDict()

        self._mustInit=True

    def isTrainable(self) :
        raise NotImplemented("Must be implemented in child")

    def getParameters(self) :
        raise NotImplemented("Must be implemented in child")

    def addHyperParameters(self, dct) :
        """adds to the list of hyper params, dct must be a dict"""
        self.hyperParameters.update(dct)

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

    def getHyperParameters(k, v) :
        """return all hyper parameter"""
        return self.hyperParameters

    def toDictionary(self) :
        """A dct representation of the object"""
        res = {
            "name": str(self.name),
            "hyperParameters": OrderedDict(self.hyperParameters),
            "notes": OrderedDict(self.notes),
        }
        
        return res

    def __repr__(self) :
        return "< %s, hps: %s >" % (self.__class__.__name__, dict(self.hyperParameters))

class UntrainableAbstraction_ABC(Abstraction_ABC):
    """docstring for UntrainableAbstraction_ABC"""
    
    def isTrainable(self) :
        return False

    def getParameters(self) :
        return {}

class TrainableAbstraction_ABC(Abstraction_ABC):
    """docstring for TrainableAbstraction_ABC"""
    def __init__(self, initializations=[], learningScenari=[], regularizations=[], **kwargs):
        super(TrainableAbstraction_ABC, self).__init__(**kwargs)

        self.abstractions={
            "initializations": initializations,
            "learningScenari": learningScenari,
            "regularizations": regularizations,
        }

        self.parameters = OrderedDict()

    def getAbstractions(self) :
        res = []
        for absts in self.abstractions.itervalues() :
            for ab in absts :
                res.append(ab)
        return res

    def isTrainable(self) :
        return True

    def setParameter(k, v) :
        """Brutally set the value of a parameter. No checks applied"""
        self.setP(k, v)
    
    def addParameters(self, dct) :
        for k, v in dct.iteritems() :
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

    def getParameters(self) :
        """return all parameter"""
        return self.parameters

    def _getParameterShape_abs(self, param, parent=None) :
        if param not in self.parameters :
            raise ValueError("Unknown parameter: %s for %s" % (param, self))
        return self.getParameterShape_abs(param, parent=None)

    def getParameterShape_abs(self, param, parent=None) :
        """Should return the shape of the parameter. This has to be implemented in order for the initializations to work (and maybe some other stuff as well)"""
        raise NotImplemented("Should be implemented in child")

    def _parametersSanityCheck(self) :
        "perform basic parameter checks on layers, automatically called on initialization"
        for k, v in self.getParameters().iteritems() :
            if not v.isSet() :
                raise ValueError("Parameter '%s' of '%s' has not been initialized" % (k, self.name) )
        
    def _initParameters(self, forceReset=False) :
        """creates the parameters if necessary"""
        if self._mustInit or forceReset :
            for init in self.abstractions["initializations"] :
                init._apply(self)
        self._mustInit=False

    def toDictionary(self, parent=None) :
        """A dct representation of the object"""
        
        res = super(TrainableAbstraction_ABC, self).toDictionary()
        ps = OrderedDict()
        for k, v in self.parameters.iteritems() :
            ps[k] = {"shape": self.getParameterShape_abs(k, parent=parent)}

        res["parameters"] = ps    
        
        return res

    def __repr__(self) :
        return "< %s, hps: %s, ps: %s >" % (self.__class__.__name__, self.hyperParameters, self.parameters)

class Apply_ABC(object):
    """Interface for abstractions that are applyied to other abstractions (all but layers)"""

    def __init__(self, **kwargs):
        super(Apply_ABC, self).__init__()
        self.name = self.__class__.__name__
        self._mustInit=True

    def logApply(self, layer, **kwargs) :
        message = "Applying : '%s' on layer '%s'" % (self.name, layer.name)
        self.logEvent(message)
        
    def _apply(self, layer, **kwargs) :
        """does self.set() + self.apply()"""
        self.logApply(layer)
        self.apply(layer, **kwargs)

    def apply(self, layer, **kwargs) :
        """Apply to a layer, basically logs stuff and then calls run"""
        raise NotImplemented("Must be implemented in child")
    
    def run(self, **kwargs) :
        """the actual worker function that does whaters the abstraction is supposed to do"""
        raise NotImplemented("Must be implemented in child")
