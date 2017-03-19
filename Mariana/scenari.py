import theano, numpy
import theano.tensor as tt
from Mariana.abstraction import Abstraction_ABC
from collections import OrderedDict

__all__ = ["LearningScenario_ABC", "ParameterGradUpdates", "OptimizerResult", "Fixed", "GradientDescent"]

class ParameterGradUpdates(object):
    """docstring for ParameterGradUpdates"""
    def __init__(self, parameter, name, update, gradient):
        super(ParameterGradUpdates, self).__init__()
        self.name = name
        self.parameter = parameter
        self.update = update
        self.gradient = gradient
        
class OptimizerFreeResults(object):
    """use this a return object for an optimizer"""
    def __init__(self):
        super(OptimizerFreeResults, self).__init__()
        self.parameters = []
    
    def add(self, parameter, name, update, gradient) :
        param = ParameterGradUpdates(parameter, name, update, gradient)
        self.parameters.append(param)

class OptimizerResult(object):
    """use this a return object for an optimizer"""
    def __init__(self, parameter, update, gradient):
        super(OptimizerResult, self).__init__()
        self.parameter = ParameterGradUpdates(parameter, "parameter", update, gradient)
        self.coParameters = []
   
    def addCoParameter(self, parameter, name, update, gradient) :
        param = ParameterGradUpdates(parameter, name, update, gradient)
        self.coParameters.append(param)
        
class LearningScenario_ABC(Abstraction_ABC):
    """
    This is the interface all scenari must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
    this class must also include a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, applyTo=None, *args, **kwargs) :
        super(Abstraction_ABC, self).__init__(*args, **kwargs)
        if applyTo :
            self.applyTo = set(applyTo)
        else :
            self.applyTo = applyTo

        self.freeParameters = OptimizerFreeResults()
        self.hyperParameters = ["applyTo"]

    def apply(self, layer, entity, paramName, loss) :
        """Apply to a layer and update networks's log"""
        hyps = {}
        for k in self.hyperParameters :
            hyps[k] = getattr(self, k)

        message = "%s uses optimizer %s of layer %s" % (entity, self.__class__.__name__, layer.name)
        layer.network.logLayerEvent(layer, message, hyps)

        if self.applyTo is not None and paramName is not self.applyTo :
            return None

        try:
            param = entity.getParameterDict()[paramName]
        except KeyError as e:
            raise KeyError("%s has no parameter %s"%(entity, paramName))

        return self.getUpdates(param, loss, layer, paramName)

    def getUpdates(self, param, loss, layer, paramName) :
        """return the updates for the parameters of layer. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Fixed(LearningScenario_ABC):
    "No learning, the layer weights stay fixed"
    def __init__(self, **kwargs):
       super(Fixed, self).__init__(**kwargs)
        
    def getUpdates(self, parameter, loss, layer) :
        ret = OptimizerResult(None, None)
        return ret

class GradientDescent(LearningScenario_ABC):
    "The GradientDescent scenario has a fixed learning rate."
    def __init__(self, lr, momentum = 0, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.hyperParameters.append("lr")
        self.hyperParameters.append("momentum")

        self.parameters = {}

    def getUpdates(self, param, loss, layer, paramName) :
        if self.momentum > 0 :
            gparam = tt.grad(loss, param)
            momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable, name="momentum.%s.%s" % (layer.name, paramName))
            param_update = self.momentum * momentum_param + (1-self.momentum)*gparam
            
            momentum_update = param - self.lr * momentum_param
            
            ret = OptimizerResult(param, param_update, gparam)
            ret.addCoParameter(momentum_param, "momentum", momentum_update, None)
        else :
            gparam = tt.grad(loss, param)
            update = param -self.lr * gparam
            ret = OptimizerResult(param, update, gparam)

        return ret