import theano, numpy
import theano.tensor as tt
from Mariana.abstraction import Abstraction_ABC
from collections import OrderedDict

__all__ = ["LearningScenario_ABC", "Fixed", "GradientDescent", "MomentumGradientDescent"]
        
class LearningScenario_ABC(Abstraction_ABC):
    """
    This is the interface all scenari must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
    this class must also include a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters.
    """
    def __init__(self, parameters=None, *args, **kwargs) :
        super(Abstraction_ABC, self).__init__(*args, **kwargs)
        if parameters :
            self.parameters = set(parameters)
        else :
            self.parameters = parameters

        self.hyperParameters = ["parameters"]

    def apply(self, layer, paramName, cost) :
        """Apply to a layer and update networks's log"""
        hyps = {}
        for k in self.hyperParameters :
            hyps[k] = getattr(self, k)

        message = "%s follows learning scenario %s" % (layer.name, self.__class__.__name__)
        layer.network.logLayerEvent(layer, message, hyps)

        if self.parameters is not None and paramName is not self.parameters :
            raise KeyError("%s is not among the list of specified parameters" % paramName)

        return self.getUpdates(layer.parameters[paramName], cost)

    def getUpdates(self, param, cost) :
        """return the updates for the parameters of layer. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Fixed(LearningScenario_ABC):
    "No learning, the layer weights stay fixed"
    def __init__(self, **kwargs):
       super(Fixed, self).__init__(**kwargs)
        
    def getUpdates(self, parameter, cost) :
        return {"update" : parameter, "gradient": 0}

class GradientDescent(LearningScenario_ABC):
    "The GradientDescent scenario has a fixed learning rate."
    def __init__(self, lr, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)
        self.lr = lr
        self.hyperParameters.append("lr")

    def getUpdates(self, param, cost) :
        gparam = tt.grad(cost, param)
        update = param -self.lr * gparam
        return {"update" : update, "gradient": gparam}

    # def getUpdates(self, thing, cost) :
    #     parameters = self.getPs()

    #     gradients = OrderedDict()
    #     updates = OrderedDict()
    #     attrNames = []
    #     for param in parameters :
    #         gparam = tt.grad(cost, param)
    #         gradients[param] = gparam
    #         updates[param] = -self.lr * gparam
    #         attrNames.append(name)
 
    #     return {"updates" : updates, "gradients": gradients, "attribute_names": attrNames}

# class GradientDescent_bck(LearningScenario_ABC):
#     "The GradientDescent scenario has a fixed learning rate."
#     def __init__(self, lr):
#        super(GradientDescent_bck, self).__init__()
#         self.lr = lr
#         self.hyperParameters = ["lr"]
#         self.gradients = {}
#         self.updates = {}

#     def getUpdates(self, layer, cost) :
#         updates = []
#         for param in layer.getParameters() :
#             gparam = tt.grad(cost, param)
#             updates.append((param, param - self.lr * gparam))
#             self.updates[param] = param - self.lr * gparam
#             self.gradients[param] = gparam
 
#         return updates

# class MomentumGradientDescent(LearningScenario_ABC):
#     "The MomentumGradientDescent scenario has a fixed learning rate and a fixed momentum."
#     def __init__(self, lr, momentum):
#         super(MomentumGradientDescent, self).__init__()
#         self.lr = lr
#         self.momentum = momentum
#         self.hyperParameters.extend(["lr", "momentum"])

#         self.gradients = {}
#         self.updates = {}

#     def getUpdates(self, layer, cost) :
#         updates = []
#         for pname, param in layer.getParameterDict().iteritems() :
#             gparam = tt.grad(cost, param)
#             momentum_param = theano.shared(param.get_value()*0., broadcastable=param.broadcastable, name="momentum-%s_%s" % (layer.name, pname))
#             v = self.momentum * momentum_param + (1-self.momentum)*gparam
#             updates.append((momentum_param, v ))
#             updates.append((param, param - self.lr * momentum_param))

#             self.updates[param] = v
#             self.updates["momentum"] = param - self.lr * momentum_param
#             self.gradients[param] = gparam

#         return updates