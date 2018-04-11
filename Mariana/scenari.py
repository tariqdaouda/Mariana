import theano, numpy
import theano.tensor as tt
from collections import OrderedDict
import lasagne.updates as LUP

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
        if previous.gradient is not None or previous.update is not None:
            raise IncompatibleLearningScenarios("Learning scenario is incompatible with previous ones (%s)" % previous)

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
    
    def __repr__(self) :
        return "< optimizer result for: %s (id: %s)>"  % (self.parameter, id(self))

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
        # self.memory = {}

    def isInheritable(self) :
        return self.inheritable

    def apply(self, abstraction, parameterName, loss, previous=None) :
        """Apply to a abstraction and update networks's log"""

        if self.applyTo is not None and parameterName not in self.applyTo :
            return None

        try:
            parameter = abstraction.getP(parameterName)
        except :
            raise KeyError("%s has no parameter %s"%(abstraction, parameterName))

        # if (parameter in self.memory) and not self.force :
            # return self.memory[parameter]

        v = self.run(parameter=parameter, parameterName=parameterName, loss=loss, abstraction=abstraction, previous=previous)
        if previous :
            try :
                return self.conflictResolve.apply(previous, v)
            except IncompatibleLearningScenarios :
                raise IncompatibleLearningScenarios("Learning scenario: '%s' is incompatible with previous updates (abstraction: '%s', previous: '%s')" % (self.__class__.__name__, abstraction.name, previous))

        # for cp in v.coParameters :
            # abstraction.registerCoParameter(cp.name, cp)
    
        return v

    def run(self, parameter, parameterName, loss, abstraction, previous) :
        """return the updates for the parameters of abstraction. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Independent(LearningScenario_ABC):
    "Indicates that the abstraction does not inherit optimization rules form the outputs. Must be placed at the first positon of the list."
    def __init__(self):
       super(Independent, self).__init__(inheritable=False)

    def run(*args, **kwargs) :
        return None

class Fixed(LearningScenario_ABC):
    "No learning, the abstraction parameters stay fixed"
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
        super(GradientDescent, self).__init__(conflictResolve=conflictResolve, **kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "momentum": momentum,
            "reverse": reverse
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        if self.getHP("momentum") == 0 :
            if not self.getHP("reverse") :
                param_update = parameter.getVar() - self.getHP("lr") * gparam
            else : 
                param_update = parameter.getVar() + self.getHP("lr") * gparam
            
            ret = OptimizerResult(parameter.getVar(), parameterName, gparam, param_update)
        else :
            momentum_param = theano.shared(parameter.getValue()*0., broadcastable=parameter.getVar().broadcastable, name="momentum.%s" % (parameterName))
            momentum_update = self.getHP("momentum") * momentum_param + (1-self.getHP("momentum"))*gparam
            if not self.getHP("reverse") :
                param_update = parameter.getVar() - self.getHP("lr") * momentum_param
            else :
                param_update = parameter.getVar() + self.getHP("lr") * momentum_param
            
            ret = OptimizerResult(parameter.getVar(), parameterName, gparam, param_update)
            ret.addCoParameter(momentum_param, "momentum", None, momentum_update)

        return ret

SGD = GradientDescent

#class GradientClipping(LearningScenario_ABC):
#    "Clips previous update to a minimum and maximum value."
#    def __init__(self, minimum, maximum, conflictResolve=Overwrite(), **kwargs):
#        """
#        """
#        super(GradientClipping, self).__init__(conflictResolve=conflictResolve, **kwargs)
#        self.addHyperParameters({
#            "minimum": minimum,
#            "maximum": maximum,
#        })
        
#    def run(self, parameter, parameterName, loss, **kwargs) :
#        previous = kwargs["previous"]
#        previous.gradient = tt.clip(previous.gradient, self.getHP("minimum"), self.getHP("maximum"))

#        return previous

class Adam(LearningScenario_ABC):
    "The Adam. Uses lasagne as backend"
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, conflictResolve=Die(), **kwargs):
        """
        use reverse = True for gradient ascent.
        """
        super(Adam, self).__init__(**kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        updates = LUP.adam( [ gparam ], [pVar], learning_rate=self.getHP("lr"), beta1=self.getHP("beta1"), beta2=self.getHP("beta2"), epsilon=self.getHP("epsilon"))

        ret = OptimizerResult(pVar, parameterName, gparam, updates[pVar])
        i = 0
        for param, update in updates.items() :
            if param is not pVar :
                name = "%s_adam_%s" % (parameterName, i)
                ret.addCoParameter(param, name, None, update)
                i += 1

        return ret

class Adamax(LearningScenario_ABC):
    "The Adamax. Uses lasagne as backend"
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8, conflictResolve=Die(), **kwargs):
        super(Adamax, self).__init__(conflictResolve=conflictResolve, **kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        updates = LUP.adamax( [ gparam ], [pVar], learning_rate=self.getHP("lr"), beta1=self.getHP("beta1"), beta2=self.getHP("beta2"), epsilon=self.getHP("epsilon"))

        ret = OptimizerResult(pVar, parameterName, gparam, updates[pVar])
        i = 0
        for param, update in updates.items() :
            if param is not pVar :
                name = "%s_adamax_%s" % (parameterName, i)
                ret.addCoParameter(param, name, None, update)
                i += 1

        return ret

class Adadelta(LearningScenario_ABC):
    "The Adadelta. Uses lasagne as backend"
    def __init__(self, lr=1.0, rho=0.9, epsilon=1e-6, conflictResolve=Die(), **kwargs):
        super(Adadelta, self).__init__(conflictResolve=conflictResolve, **kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "rho": rho,
            "epsilon": epsilon,
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        updates = LUP.adadelta( [ gparam ], [pVar], learning_rate=self.getHP("lr"), rho=self.getHP("rho"), epsilon=self.getHP("epsilon"))

        ret = OptimizerResult(pVar, parameterName, gparam, updates[pVar])
        i = 0
        for param, update in updates.items() :
            if param is not pVar :
                name = "%s_adadelta_%s" % (parameterName, i)
                ret.addCoParameter(param, name, None, update)
                i += 1

        return ret

class Adagrad(LearningScenario_ABC):
    "The Adagrad. Uses lasagne as backend"
    def __init__(self, lr=1.0, epsilon=1e-6, conflictResolve=Die(), **kwargs):
        super(Adagrad, self).__init__(conflictResolve=conflictResolve, **kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "epsilon": epsilon,
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        updates = LUP.adagrad( [ gparam ], [pVar], learning_rate=self.getHP("lr"), epsilon=self.getHP("epsilon"))

        ret = OptimizerResult(pVar, parameterName, gparam, updates[pVar])
        i = 0
        for param, update in updates.items() :
            if param is not pVar :
                name = "%s_adagrad_%s" % (parameterName, i)
                ret.addCoParameter(param, name, None, update)
                i += 1

        return ret

class RMSProp(LearningScenario_ABC):
    "The RMSProp. Uses lasagne as backend"
    def __init__(self, lr=1.0, rho=0.9, epsilon=1e-6, conflictResolve=Die(), **kwargs):
        super(RMSProp, self).__init__(conflictResolve=conflictResolve, **kwargs)
        
        self.addHyperParameters({
            "lr": lr,
            "rho": rho,
            "epsilon": epsilon,
        })
        
    def run(self, parameter, parameterName, loss, **kwargs) :
        pVar = parameter.getVar()
        gparam = tt.grad(loss, pVar)
        updates = LUP.rmsprop( [ gparam ], [pVar], learning_rate=self.getHP("lr"), rho=self.getHP("rho"), epsilon=self.getHP("epsilon"))

        ret = OptimizerResult(pVar, parameterName, gparam, updates[pVar])
        i = 0
        for param, update in updates.items() :
            if param is not pVar :
                name = "%s_rmsprop_%s" % (parameterName, i)
                ret.addCoParameter(param, name, None, update)
                i += 1

        return ret
