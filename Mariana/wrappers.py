from collections import OrderedDict
import theano, sys, numpy
import sys

import Mariana.candies as MCAN
import Mariana.settings as MSET
import Mariana.custom_types as MTYPES

class TheanoFunctionHandle(object) :
    """
    This class encapsulates a Theano function.
    TheanoFunction objects should be defined as self attributes in the setCustomTheanoFunctions() function of output layers.
    It will also generate custom error messages whose verbosity depends on Mariana.settings.VERBOSE. Set it to False to get quieter
    error messages.
    """

    def __init__(self, name, layer, output, stream, updates={}, **theano_kwargs) :
        """
        :param Output layer: the output layer the function should be applied to
        :param list updates: boolean defining if updates should be applied
        :param dict \*\*theano_kwargs: additional arguments to passed to the real theano function underneath
        :parama stream: usually something like "train" or "test". Defines if regularizations and decorators should be applied
        """
        def _bckTrckInputs(current_layer, stream, inputs = OrderedDict()) :     
            for k, v in current_layer.__dict__.iteritems() :
                if v.__class__ is MTYPES.Inputs or v.__class__ is MTYPES.Targets :
                    inputs["%s.%s" % (current_layer.name, k)] = v[stream]
  
            for layer in current_layer.network.inConnections[current_layer] :
                _bckTrckInputs(layer, stream, inputs)
            
            return inputs

        self.name = name
        self.updates = updates
        self.layer = layer
        self.stream = stream
        self.theano_kwargs = theano_kwargs

        self.inputs = _bckTrckInputs(layer, stream)
            
        self.output = output

        self.theano_fct = None

    def __add__(self, other) :
        if other.__class__ is TheanoFunction :
            if other.isCompiled() :
                raise TypeError("Added value cannot be an already compiled function")
            other._addHandle(self)
            return other
        elif other.__class__ is not TheanoFunctionHandle :
            raise TypeError("Added value must be another valid function")

        return TheanoFunction([self, other])

    def __call__(self, *args, **kwargs) :
        if not self.theano_fct :
            self.theano_fct = TheanoFunction([self])
            
        return self.theano_fct.run(*args, **kwargs)

class TheanoFunction(object) :

    def __init__(self, theano_handles) :

        self.theano_handles = theano_handles
        self.theano_fct = None

    def isCompiled(self) :
        return self.theano_fct is not None

    def _addHandle(self, theano_handle) :
        self.theano_handles.append(theano_handle)
    
    def _compile(self):

        if not self.isCompiled() :
            self.inputs = OrderedDict()
            self.outputs = OrderedDict()
            self.updates = OrderedDict()
            
            self.fct_inputs_varToName = OrderedDict()
            self.fct_inputs = OrderedDict()
            
            self.theano_kwargs = {}
            for handle in self.theano_handles :
                self.theano_kwargs.update(handle.theano_kwargs)
                
                i = 0
                for k, v in handle.inputs.iteritems() :
                    self.inputs[k] = v
                    self.fct_inputs_varToName[v] = k
                    self.fct_inputs[k] = None
                    i += 1

                self.outputs["%s.%s" % (handle.layer.name, handle.name)] = handle.output
                for param, update in handle.updates.iteritems() :
                    try:
                        self.updates[param] += update
                    except KeyError as e:
                        self.updates[param] = param + update

            self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.outputs.values(), updates = self.updates, **self.theano_kwargs)

            if MSET.DEVICE_IS_GPU :
                if str(self.getToposort()).find("float64") > -1:
                    msg = "There are some float64s that do not fit on the GPU and will slow down the computations.\nPlease consider:"
                    msg += "\n\t* Launching with THEANO_FLAGS=device=gpu,floatX=float32 python <your script>.py."
                    msg += "\n\t* If you have any dmatrix, dvector or dscalar in your code replace them with matrix, vector, scalar."
                    MCAN.friendly("Run device", msg, warning = True)
            
            self.results = OrderedDict()

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

    def run(self, inputs = {}) :
        """Run the theano function with the kwargs. Will return an OrderedDict of the outputs"""
        # def _die(fctName, layer, kwargs, exc) :
        #     localStr = "!!=> Error in function '%s' for layer '%s':\n%s\n" % (fctName, layer.name, exc.message)
        #     sys.stderr.write(localStr)
        #     sys.stderr.write("Have a look at the log file: %s for details about the arguments" % MSET.SAVE_MESSAGE_LOG_FILE)
        #     strArgs = []
        #     for k, v in kwargs.iteritems() :
        #         strArgs.append("%s, shape: %s \n----\n%s" % (k, numpy.asarray(v).shape, v))
        #     MCAN.fatal(localStr, "!!=> the arguments were:\n %s\n" % ('\n'.join(strArgs)), toRaise = exc)

        self._compile()
        nbArgs = 0
        for k, v in inputs.iteritems() :
            try :
                self.fct_inputs[k] = inputs[k]
            except KeyError :
                try:
                    kk = self.fct_inputs_varToName[k]
                    self.fct_inputs[kk] = inputs[v]
                except KeyError as e:
                    raise TypeError("'%s' is not among this function arguments" % k)
            nbArgs += 1

        if nbArgs != len(self.fct_inputs) :
            args = str(self.fct_inputs.keys()).replace("[", "").replace("]", "")
            raise TypeError("Function expects the following arguments: %s, %d provided" % (args, nbArgs) )

        fres = iter(self.theano_fct(*self.fct_inputs.values()))
        # try :
        #     fres = iter(self.theano_fct(*self.fct_inputs.values()))
        # except Exception as e:
        #     _die(self.name, self.layer, kwargs, e)
    
        for k in self.outputs.iterkeys() :
            self.results[k] = fres.next()

        return self.results

    def __call__(self, *args, **kwargs) :
        return self.run(*args, **kwargs)

    def __repr__(self) :
        return "<Mariana Theano Fct '%s'>" % self.name

    def __str__(self) :
        return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)
