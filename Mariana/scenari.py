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
    def __init__(self, parameter, name, gradient, update):
        super(ParameterGradUpdates, self).__init__()
        self.name = name
        self.parameter = parameter
        self.update = update
        self.gradient = gradient

class OptimizerResult(ParameterGradUpdates):
    """use this a return object for an optimizer"""
    def __init__(self, parameter, name, gradient, update):
        super(OptimizerResult, self).__init__(parameter, name, gradient, update)
        self.coParameters = []
   
    def addCoParameter(self, parameter, name, gradient, update) :
        param = ParameterGradUpdates(parameter, name, gradient, update)
        self.coParameters.append(param)
        
class LearningScenario_ABC(MABS.UntrainableAbstraction_ABC, MABS.Apply_ABC):
    """
    This is the interface all optimizations rules must expose.
    """
    def __init__(self, applyTo=None, inheritable=True, conflictResolve=Die(), **kwargs) :
        super(LearningScenario_ABC, self).__init__(**kwargs)
        if applyTo :
            self.applyTo = set(applyTo)
            self.setHP("applyTo", applyTo)
        else :
            self.applyTo = applyTo

        self.inheritable = inheritable
        self.conflictResolve = conflictResolve

    def apply(self, abstraction, parameterName, loss, previous=None) :
        """Apply to a abstraction and update networks's log"""

        if self.applyTo is not None and parameterName not in self.applyTo :
            return None

        try:
            parameter = abstraction.getP(parameterName)
        except :
            raise KeyError("%s has no parameter %s"%(abstraction, parameterName))

        v = self.run(parameter=parameter, parameterName=parameterName, loss=loss, abstraction=abstraction, previous=previous)
        if previous :
            try :
                return self.conflictResolve.apply(previous, v)
            except IncompatibleLearningScenarios :
                raise IncompatibleLearningScenarios("Learning scenario: '%s' is incompatible with previous updates (abstraction: '%s')" % (self.__class__.__name__, abstraction.name))
        return v

    def run(self, parameter, parameterName, loss, abstraction, previous) :
        """return the updates for the parameters of abstraction. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Fixed(LearningScenario_ABC):
    "No learning, the abstraction parameteres stay fixed"
    def __init__(self, applyTo=None, inheritable=False, conflictResolve=Overwrite(), **kwargs):
       super(Fixed, self).__init__(applyTo, inheritable, conflictResolve, **kwargs)
        
    def run(self, parameter, **kwargs) :
        ret = OptimizerResult(parameter, None, None, None)
        return ret

class GradientDescent(LearningScenario_ABC):
    "The GradientDescent scenario has a fixed learning rate."
    def __init__(self, lr, momentum=0, reverse=False, conflictResolve=Die(), **kwargs):
        """
        use reverse = True for gradient ascent.
        """
        super(GradientDescent, self).__init__(**kwargs)
        
        self.addHyperParameters({
        	"lr": lr,
        	"momentum": momentum,
        	"reverse": reverse
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        gparam = tt.grad(loss, parameter.getVar())
        if self.getHP("momentum") <= 0 :
            param_update = parameter.getVar() - self.getHP("lr") * gparam
            if self.getHP("reverse") :
                param_update = -param_update
            ret = OptimizerResult(parameter.getVar(), parameterName, gparam, param_update)
        else :
            momentum_param = theano.shared(parameter.getVar()*0., broadcastable=parameter.broadcastable, name="momentum.%s" % (parameterName))
            momentum_update = self.momentum * momentum_param + (1-self.getHP("momentum"))*gparam
            param_update = parameter.getVar() + self.getHP("lr") * momentum_param
            if self.getHP("reverse") :
                param_update = -param_update
            ret = OptimizerResult(parameter.getVar(), parameterName, gparam, param_update)
            ret.addCoParameter(momentum_param, "momentum", None, momentum_update)

        return ret
