import theano, numpy
import theano.tensor as tt
from collections import OrderedDict

import Mariana.abstraction as MABS


__all__ = ["LearningScenario_ABC", "ParameterGradUpdates", "OptimizerFreeResults", "OptimizerResult", "Fixed", "GradientDescent"]

class IncompatibleLearningScenarios(Exception) :
    def __init__(self, msg) :
        self.message = msg

    def __str__(self) :
        return self.message
        
    def __repr__(self):
        return self.message

class ConflictResolve(object):
    """In Mariana scenari can be chained. The children of that class defines what to do in case of conflict"""
    def __init__(self, warning=False):
        super(ConflictResolve, self).__init__()
        self.warning=warning

    def apply(self, previous, current) :
        if self.warning :
            print("Resolving conflict between scenari using %s" % self.__class__.__name__)
        
        return self.resolve(previous, current)

    def resolve(self, previous, current) :
        raise NotImplemented("Should be implemented in child")

class Overwrite(ConflictResolve):
    """Overwrite the previous value"""
    def resolve(self, previous, current) :
        return current

class Ignore(ConflictResolve):
    """Ignore the update and keep the previous value"""
    def resolve(self, previous, current) :
        return previous

class Die(ConflictResolve):
    """No conflic resolve, crashes everything"""
    def resolve(self, previous, current) :
        raise IncompatibleLearningScenarios("Learning scenario is incompatible with previous ones")

class ParameterGradUpdates(object):
    """docstring for ParameterGradUpdates"""
    def __init__(self, parameter, name, update, gradient):
        super(ParameterGradUpdates, self).__init__()
        self.name = name
        self.parameter = parameter
        self.update = update
        self.gradient = gradient
        
class OptimizerFreeResults(object):
    """Parameters of a scenario that need to be updated"""
    def __init__(self):
        super(OptimizerFreeResults, self).__init__()
        self.parameters = []
    
    def add(self, parameter, name, update, gradient) :
        param = ParameterGradUpdates(parameter, name, update, gradient)
        self.parameters.append(param)

class OptimizerResult(object):
    """use this a return object for an optimizer"""
    def __init__(self, parameter, parameterName, update, gradient):
        super(OptimizerResult, self).__init__()
        self.parameter = ParameterGradUpdates(parameter, "parameterName", update, gradient)
        self.coParameters = []
   
    def addCoParameter(self, parameter, name, update, gradient) :
        param = ParameterGradUpdates(parameter, name, update, gradient)
        self.coParameters.append(param)
        
class LearningScenario_ABC(MABS.ApplyAbstraction_ABC):
    """
    This is the interface all scenari must expose.
    """
    def __init__(self, applyTo=None, inheritable=True, conflictResolve=Die(), *args, **kwargs) :
        super(LearningScenario_ABC, self).__init__(*args, **kwargs)
        if applyTo :
            self.applyTo = set(applyTo)
            self.setHP("applyTo", applyTo)
        else :
            self.applyTo = applyTo

        self.inheritable = inheritable
        self.conflictResolve = conflictResolve
        self.freeParameters = OptimizerFreeResults()

    def apply(self, layer, abstraction, paramName, loss, previous=None) :
        """Apply to a layer and update networks's log"""

        if self.applyTo is not None and paramName not in self.applyTo :
            return None

        try:
            param = abstraction.getP(paramName)
        except :
            raise KeyError("%s has no parameter %s"%(abstraction, paramName))

        v = self.run(param, loss, **{"layer": layer, "paramName": paramName, "previous": previous, "abstraction": abstraction})
        if previous :
            try :
                return self.conflictResolve.apply(previous, v)
            except IncompatibleLearningScenarios :
                raise IncompatibleLearningScenarios("Learning scenario: '%s' is incompatible with previous updates (layer: '%s')" % (self.__class__.__name__, layer.name))
        return v

    def run(self, param, loss, abstraction, layer, paramName, previous) :
        """return the updates for the parameters of layer. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Fixed(LearningScenario_ABC):
    "No learning, the layer weights stay fixed"
    def __init__(self, applyTo=None, inheritable=False, conflictResolve=Overwrite(), **kwargs):
       super(Fixed, self).__init__(applyTo, inheritable, conflictResolve, **kwargs)
        
    def run(self, param, **kwargs) :
        ret = OptimizerResult(param, None, None, None)
        return ret

class GradientDescent(LearningScenario_ABC):
    "The GradientDescent scenario has a fixed learning rate."
    def __init__(self, lr, momentum=0, reverse=False, **kwargs):
        """
        use reverse = True for gradient ascent.
        """
        super(GradientDescent, self).__init__(**kwargs)
        
        self.addHyperParameters({
        	"lr": lr,
        	"momentum": momentum,
        	"reverse": reverse
        })
        
    def run(self, param, loss, paramName, **kwargs) :
        gparam = tt.grad(loss, param.getVar())
        if self.getHP("momentum") <= 0 :
            param_update = param.getVar() - self.getHP("lr") * gparam
        else :
            momentum_param = theano.shared(param.getVar()*0., broadcastable=param.broadcastable, name="momentum.%s" % (paramName))
            momentum_update = self.momentum * momentum_param + (1-self.getHP("momentum"))*gparam
            param_update = param.getVar() + self.getHP("lr") * momentum_param
            ret.addCoParameter(momentum_param, "momentum", momentum_update, None)

        if self.getHP("reverse") :
            param_update = -param_update

        ret = OptimizerResult(param.getVar(), paramName, param_update, gparam)
        return ret
