import theano
import theano.tensor as tt
import Mariana.abstraction as MABS

__all__ = ["Cost_ABC", "Null", "NegativeLogLikelihood", "MeanSquaredError", "CrossEntropy", "CategoricalCrossEntropy", "BinaryCrossEntropy"]

class Cost_ABC(MABS.UntrainableAbstraction_ABC, MABS.Apply_ABC) :
    """This is the interface a Cost must expose. In order for the trainer/recorder to know which attributes are hyper-parameters,
    this class must also include a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
    as hyper-parameters."""

    def __init__(self, reverse=False, streams=["test", "train"], **kwargs) :
        """use reverse = True, to have the opposite of cost"""
        super(Cost_ABC, self).__init__(streams=streams, **kwargs)
        self.setHP("reverse", reverse)
       
    def logApply(self, layer, **kwargs) :
        message = "Applying '%s' on layer '%s'" % (self.name, self.getHP('parameter'), layer.name)
        if self.getHP("reverse") :
            message += " (reverse)"
        self.logEvent(message)

    def apply(self, layer, targets, outputs, stream) :
        """Apply to a layer and update networks's log"""

        if self.getHP("reverse") :
            return -self.run(targets, outputs, stream)
        else :
            return self.run(targets, outputs, stream)

    def run(self, targets, outputs, stream) :
        """The cost function. Must be implemented in child"""
        raise NotImplemented("Must be implemented in child")

class Null(Cost_ABC) :
    """No cost at all"""
    def run(self, targets, outputs, stream) :
        return tt.sum(outputs*0 + targets*0)

class NegativeLogLikelihood(Cost_ABC) :
    """For a probalistic output, works great with a softmax output layer"""
    def run(self, targets, outputs, stream) :
        cost = -tt.mean(tt.log(outputs)[tt.arange(targets.shape[0]), targets])
        return cost

class MeanSquaredError(Cost_ABC) :
    """The all time classic"""
    def run(self, targets, outputs, stream) :
        cost = tt.mean((outputs - targets) ** 2)
        return cost

class AbsoluteAverage(Cost_ABC) :
    """Average absolute value"""
    def run(self, targets, outputs, stream) :
        cost = tt.mean((outputs - targets))
        return cost

class CategoricalCrossEntropy(Cost_ABC) :
    """Returns the average number of bits needed to identify an event."""
    def run(self, targets, outputs, stream) :
        cost = tt.mean( tt.nnet.categorical_crossentropy(outputs, targets) )
        return cost
      
CrossEntropy = CategoricalCrossEntropy
# class CrossEntropy(CategoricalCrossEntropy) :
#     """Short hand for CategoricalCrossEntropy"""
#     pass

class BinaryCrossEntropy(Cost_ABC) :
    """Use this one for binary data"""
    def run(self, targets, outputs, stream) :
        cost = tt.mean( tt.nnet.binary_crossentropy(outputs, targets) )
        return cost