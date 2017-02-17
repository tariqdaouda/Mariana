from collections import OrderedDict
import theano, sys, numpy
import sys

import Mariana.candies as MCAN
import Mariana.settings as MSET

class TheanoFunction(object) :
    """
    This class encapsulates a Theano function.
    TheanoFunction objects should be defined as self attributes in the setCustomTheanoFunctions() function of output layers.
    It will also generate custom error messages whose verbosity depends on Mariana.settings.VERBOSE. Set it to False to get quieter
    error messages.
    """

    def __init__(self, name, layer, output_expressions, additional_input_expressions = {}, updates = [], **kwargs) :
        """
        :param str name: name of the function
        :param Output layer: the output layer the function should be applied to
        :param list output_expressions: list of tuples of symbolic expressions you want as output and the names you want to give to them: (name, expressions)
        :param dict additional_input_expressions: additional inputs needed to compute the expressions
        :param list updates: list of tuples (shared variable, symbolic expression of the update to be applied to it)
        :param dict \*\*kwargs: additional arguments to passed to the real theano function underneath
        """
        def _bckTrckInputs(startLayer, inputs = OrderedDict(), inpSet = set()) :        
            if MSET.TYPE_INPUT_LAYER in startLayer.types :
                inpOut = startLayer.inputs
                if inpOut not in inpSet :
                    inputs[startLayer.name] = inpOut
                    inpSet.add(inpOut)
            
            for layer in startLayer.network.inConnections[startLayer] :
                _bckTrckInputs(layer, inputs, inpSet)
            
            return inputs, inpSet

        self.cast_warning_told = False

        self.name = name
        self.layer = layer

        self.inputs, inpSet = _bckTrckInputs(layer)

        for k, v in additional_input_expressions.iteritems() :
            if v not in inpSet :
                self.inputs[k] = v
                inpSet.add(v)

        self.fctInputs = OrderedDict()
        for i in self.inputs :
            self.fctInputs[i] = None

        self.additional_input_expressions = additional_input_expressions
        
        self.output_expressions = OrderedDict()
        for name, output_expr in output_expressions :
            self.output_expressions[name] = output_expr
        
        kwUpdates = {}
        for k, v in updates :
            if k in kwUpdates :
                message = "Parameter '%s' has more than one defined update, only using the first" % k
                if MSET.VERBOSE :
                    print(message)
                layer.network.logLayerEvent(layer, message)     
            else :
                kwUpdates[k] = v

        self.updates = kwUpdates.items()
        # print self.name, self.inputs, layer, self.output_expressions
        self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.output_expressions.values(), updates = self.updates, **kwargs)

        warningMsg = False
        if MSET.DEVICE_IS_GPU :
            device = "GPU"
            msg = "I will use the [-%s-] to run function '%s' of layer '%s'!" % (device, name, layer.name)
            if str(self.getToposort()).find("float64") > -1:
                warningMsg = True
                msg += "\n\nBut there are some float64s that do not fit on the GPU and will slow down the computations.\nPlease consider:"
                msg += "\n\t* Launching with THEANO_FLAGS=device=gpu,floatX=float32 python <your script>.py."
                msg += "\n\t* If you have any dmatrix, dvector or dscalar in your code replace them with matrix, vector, scalar."
        else:
            device = "CPU"
            msg = "I will use the [-%s-] to run function '%s' of layer '%s'!" % (device, name, layer.name)

        self.results = OrderedDict()
        MCAN.friendly("Run device", msg, warning = warningMsg)

    def getToposort(self) :
        """returns the toposort ( name of all ops  of the function in order of application ) of the function"""
        return self.theano_fct.maker.fgraph.toposort()

    def printGraph(self) :
        """Print the theano graph of the function"""
        theano.printing.debugprint(self.theano_fct)

    def dumpToImage(self, filename) :
        """saves the function graph into an image file, requires pydot"""
        theano.printing.pydotprint(self.theano_fct, filename)

    def dumpToHTML(self, filename) :
        """saves the function graph into an interactive html file, requires pydot and potentially an internet connection"""
        import theano.d3viz as d3v
        d3v.d3viz(self.theano_fct, filename)

    def run(self, **kwargs) :
        """Run the theano function with the kwargs. Will return an OrderedDict of the outputs"""
        def _die(fctName, layer, kwargs, exc) :
            localStr = "!!=> Error in function '%s' for layer '%s':\n%s\n" % (fctName, layer.name, exc.message)
            sys.stderr.write(localStr)
            sys.stderr.write("Have a look at the log file: %s for details about the arguments" % MSET.SAVE_MESSAGE_LOG_FILE)
            strArgs = []
            for k, v in kwargs.iteritems() :
                strArgs.append("%s, shape: %s \n----\n%s" % (k, numpy.asarray(v).shape, v))
            MCAN.fatal(localStr, "!!=> the arguments were:\n %s\n" % ('\n'.join(strArgs)), toRaise = exc)

        self.fctInputs.update(kwargs)
        try :
            fres = iter(self.theano_fct(*self.fctInputs.values()))
        except Exception as e:
            _die(self.name, self.layer, kwargs, e)
            
        for k in self.output_expressions.iterkeys() :
            self.results[k] = fres.next()

        return self.results

    def __call__(self, **kwargs) :
        return self.run(**kwargs)

    def __repr__(self) :
        return "<Mariana Theano Fct '%s'>" % self.name

    def __str__(self) :
        return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)
