from collections import OrderedDict
import theano
import theano.tensor as tt

import Mariana.candies as MCAN
import Mariana.settings as MSET
import Mariana.custom_types as MTYPES

__all__=["UpdateStore", "Updates", "TheanoFunctionHandle", "TheanoFunction"]

class UpdateStore(object):
    """Stores gradients and updates for parameters in convenient interface"""
    def __init__(self):
        super(UpdateStore, self).__init__()
        self.names = {
            "updates": OrderedDict(),
            "gradients": OrderedDict(),
        }
        self.updates = OrderedDict()
        self.gradients = OrderedDict()
    
    def add(self, parameter, gradient, update, name) :
        if update is None: 
            if parameter in self.updates :
                del self.updates[parameter]
                del self.names["updates"][parameter]
        else :
            self.updates[parameter] = update
            self.names["updates"][parameter] = name
            
        if gradient is None:
            if parameter in self.gradients:
                del self.gradients[parameter]
                del self.names["gradients"][parameter]
        else :
            self.gradients[parameter] = gradient
            self.names["gradients"][parameter] = name

class Updates(object):
    """Derives the updates for a theano function"""
    def __init__(self, output_layer, stream):
        super(Updates, self).__init__()
        self.output_layers = [output_layer]
        self.loss = 0
        self.stream = stream
        self.store = UpdateStore()
        self.isCompiled = False

    def merge(self, updates) :
        """Merge tow updates by combining their outputs and adding theirs losses. Must be None or another Updates instance"""
        if self.isCompiled :
            raise ValueError("Impossible to merge with already compiled updates")

        if updates is not None :
            if not isinstance(updates, self.__class__) :
                raise ValueError("Parameter must be an instance of '%s" % self.__class__)
            self.output_layers.extend(updates.output_layers)
            # self.loss += updates.loss

    def compile(self) :
        """Derive the updates and gradients for every parameter in the network"""
        
        # def _apply(sc, store, layer, entity, param_name, cute_name, loss) :
        #     try :
        #         previous = store[cute_name]
        #     except KeyError :
        #         previous = None

        #     store[cute_name] = sc.apply(layer, entity, param_name, loss, previous)
        
        #append outputs optimization rules at the top level
        self.loss = 0
        optimizers = {}
        names = {}
        for o in self.output_layers :
            self.loss += o.loss[self.stream]
            optimizers[o] = o.abstractions["learningScenari"]
            names[o] = o.name
            for abstraction in o.getTrainableAbstractions() :
                optimizers[abstraction] = o.abstractions["learningScenari"]
                names[abstraction] = "%s.%s" % (o.name. abstraction.name)
            
            for l in o.dependencies.itervalues() :
                names[l] = l.name
                try :
                    optimizers[l].extend(o.abstractions["learningScenari"])
                except KeyError :
                    optimizers[l] = o.abstractions["learningScenari"]
    
                for abstraction in l.getTrainableAbstractions() :
                    names[abstraction] = "%s.%s" % (l.name. abstraction.name)
                    try :
                        optimizers[abstraction].extend(o.abstractions["learningScenari"])
                    except KeyError :
                        optimizers[abstraction] = o.abstractions["learningScenari"]

        #append specific optimization rules
        for abstraction in optimizers :
            if abstraction not in self.output_layers :
                optimizers[abstraction].extend(abstraction.abstractions["learningScenari"])
                for reg in abstraction.abstractions["regularizations"] :
                    self.loss = reg.apply(abstraction, self.loss)

        for abstraction, scenari in optimizers.iteritems() :
            for paramName in abstraction.getParameters() :
                previousOptim=None
                for sc in scenari :
                    optim = sc.apply(abstraction=abstraction, parameterName=paramName, loss=self.loss, previous=previousOptim)
                    self.store.add( optim.parameter, optim.gradient, optim.update, "%s.%s" % (names[abstraction], paramName) )
                    for coParam in optim.coParameters :
                        self.store.add( coParam.parameter, coParam.gradient, coParam.update, "%s.%s.%s" % (names[abstraction], paramName, coParam.name) )
                    previousOptim = optim
        
        self.isCompiled = True

class TheanoFunctionHandle(object) :
    """
    This a description of a theano function. It is used as a reference for just-in-time compilation of theano functions.
    TheanoFunctionHandle can be added to each other to create complex rich functions.
    """

    def __init__(self, name, layer, output, stream, update=False, **theano_kwargs) :
        """
        :param str name: The name of the function used to identify its return value 
        :param output: The theano expression to be returned
        :parama stream: Usually something like "train" or "test". Defines if regularizations and decorators should be applied
        :parama bool update: Should it update the parameters when called
        :param dict \*\*theano_kwargs: additional arguments to passed to the real theano function underneath
        """
        super(TheanoFunctionHandle, self).__init__()
        
        def _bckTrckInputs(current_layer, stream, inputs = OrderedDict()) :     
            for k, v in current_layer.__dict__.iteritems() :
                if v.__class__ is MTYPES.Inputs :
                    inputs["%s.%s" % (current_layer.name, k)] = v[stream]
  
            for layer in current_layer.network.inConnections[current_layer] :
                _bckTrckInputs(layer, stream, inputs)
            
            return inputs

        self.name = name
        self.update = update
        self.layer = layer
        self.stream = stream
        self.theano_kwargs = theano_kwargs

        self.inputs = _bckTrckInputs(layer, stream)
        for k, v in layer.__dict__.iteritems() :
            if v.__class__ is MTYPES.Targets :
                self.inputs["%s.%s" % (layer.name, k)] = v[stream]

        try :
            self.output = output[stream]
        except :
            raise ValueError("Output does not have a stream: '%s'" % stream)

        # self.theano_fct = None

    def hasUpdates(self) :
        """returns True if the function updates the parameters, False other wise."""
        return self.update

    # def __add__(self, other) :
    #     """Add two handles together using the '+' operator. Losses will be added up and updates re-calculated"""
    #     if self.stream != other.stream :
    #         raise TypeError("All functions must be in the same stream %s != %s" %(self.stream, other.stream))
        
    #     if other.__class__ is TheanoFunction :
    #         if other.isCompiled() :
    #             raise TypeError("Cannot add an already compiled function")
    #         other._addHandle(self)
    #         return other
    #     elif other.__class__ is not TheanoFunctionHandle :
    #         raise TypeError("Added value must be another valid function")

    #     return TheanoFunction([self, other])

    # def develop(self) :
    #     """Compile the inner theano function"""
    #     if not self.theano_fct :
    #         self.theano_fct = TheanoFunction([self])

    # def __call__(self, *args, **kwargs) :
    #     """Call the inner theano functiom"""
    #     self.develop()
    #     return self.theano_fct.run(*args, **kwargs)

    # def __getattribute__(self, k) :
    #     """return the theano function attributes"""
    #     self.develop()
    #     if hasattr(self.theano_fct, k) :
    #         return getattr(self.theano_fct, k)
    
class TheanoFunction(object) :
    """
    This class encapsulates a just-in-time compiled Theano function.
    """
    def __init__(self, name, layer, output, stream, update=False, **theano_kwargs) :
        super(TheanoFunction, self).__init__()

        self.theano_handles = [TheanoFunctionHandle(name, layer, output, stream, update=update, **theano_kwargs)]
        self.theano_fct = None
        self.gradients_fct = None
        self.updates_fct = None

        self.inputs = None
        self.outputs = None
        self.updates = None
    
        self.perfomUpdates = None
            
    def isCompiled(self) :
        """Has the compildation already happend?"""
        return self.theano_fct is not None

    def __add__(self, theano_handle) :
        self._addHandle(theano_handle)
    
    def _addHandle(self, theano_handle) :
        """Add a handle to the current definition. Losses will be added up and updates will be merged"""
        if not isinstance(theano_handle, TheanoFunctionHandle) :
            raise ValueError("theano_handle must be an instance of TheanoFunctionHandle")

        self.theano_handles.append(theano_handle)
    
    def compile(self):
        """Compile the function just-in-time according the definitions given by the handles."""
        if not self.isCompiled() :
            self.perfomUpdates = False
            
            self.inputs = OrderedDict()
            self.inputs_varToName = OrderedDict()
            self.outputs = OrderedDict()
            self.updates = None
            
            self.theano_kwargs = {}
            varToName = OrderedDict()
            for handle in self.theano_handles :
                self.theano_kwargs.update(handle.theano_kwargs)
                
                for k, v in handle.inputs.iteritems() :
                    varToName[v] = k
                    
                self.outputs["%s.%s" % (handle.layer.name, handle.name)] = handle.output
                
                if handle.hasUpdates() :
                    self.perfomUpdates = True
                    if self.updates is None :
                        self.updates = Updates(handle.layer, handle.stream)
                    else :
                        self.updates.merge(Updates(handle.layer, handle.stream))

            all_theano_inputs = set(theano.gof.graph.inputs(self.outputs.values()))
            for inp in all_theano_inputs :
                try :
                    name = varToName[inp]
                    self.inputs[name] = inp
                    self.inputs_varToName[inp] = name
                except KeyError :
                    pass

            updates = {}
            if self.updates :
                self.updates.compile()
                if self.perfomUpdates :
                    updates = self.updates.store.updates
            
            self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.outputs.values(), updates = updates, **self.theano_kwargs)
            
            # if MSET.DEVICE_IS_GPU :
            #     if str(self.getToposort()).find("float64") > -1:
            #         msg = "There are some float64s that do not fit on the GPU and will slow down the computations.\nPlease consider:"
            #         msg += "\n\t* Launching with THEANO_FLAGS=device=gpu,floatX=float32 python <your script>.py."
            #         msg += "\n\t* If you have any dmatrix, dvector or dscalar in your code replace them with matrix, vector, scalar."
            #         MCAN.friendly("Run device", msg, warning = True)
            
            self.results = OrderedDict()

    def _parseInputs(self, inputs = {}) :
        """parse function inputs and raises SyntaxError exceptions with friendly, human readable errors"""
        fct_inputs = OrderedDict()
        if len(inputs) != len(self.inputs_varToName) :
            givens = set(inputs.keys())
            expected = set(self.inputs_varToName.values())
            missing = list(expected - givens)
            notInvited = list(givens - expected)
            msg = []
            if len(missing) > 0 :
                msg.append("Missing arguments: %s" % str(missing)[1:-1])
            if len(notInvited) > 0 :
                msg.append("Unexpected arguments: %s" % str(notInvited)[1:-1])
            if len(msg) > 0 :
                raise SyntaxError('\n'.join(msg))
  
        for param, pname in self.inputs_varToName.iteritems() :
            fct_inputs[param] = inputs[pname]
        
        return fct_inputs

    def run(self, inputs = {}) :
        """run the function and return the results"""
        self.compile()
        fct_inputs = self._parseInputs(inputs)

        fres = iter(self.theano_fct(*fct_inputs.values()))
        for k in self.outputs.iterkeys() :
            self.results[k] = fres.next()

        return self.results

    def getGradients(self, inputs={}) :
        """return the gradients that would be performed"""
        self.compile()
        if not self.updates :
            raise TypeError("Function has no updates, cannot have gradients")

        fct_inputs = self._parseInputs(inputs)
        if not self.gradients_fct :
            self.gradients_fct = theano.function(inputs = self.inputs.values(), outputs = self.updates.store.gradients.values(), **self.theano_kwargs)

        fres = iter(self.gradients_fct(*fct_inputs.values()))
        results = OrderedDict()
        for k in self.updates.store.names["gradients"].itervalues() :
            results[k] = fres.next()

        return results
        
    def getUpdates(self, inputs={}) :
        """return the updates that would be performed"""
        self.compile()
        if not self.updates :
            raise TypeError("Function has no updates")
        fct_inputs = self._parseInputs(inputs)

        if not self.updates_fct :
            self.updates_fct = theano.function(inputs = self.inputs.values(), outputs = self.updates.store.updates.values(), **self.theano_kwargs)

        fres = iter(self.updates_fct(*fct_inputs.values()))
        results = OrderedDict()
        for k in self.updates.store.names["updates"].itervalues() :
            results[k] = fres.next()

        return results

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

    def __call__(self, *args, **kwargs) :
        """a convenient alias to run()"""
        return self.run(*args, **kwargs)

    def __repr__(self) :
        if not self.isCompiled :
            return "<Uncompiled Mariana Theano Fct '%s'>" % id(self)
        else :
            args = [] 
            for k, v in self.inputs_varToName.items() :
                args.append("%s: %s" %(v, k))

            return "<Compiled Mariana Theano Fct '%s'. Arguments: '%s'>" % (id(self), ', '.join(args))

class TheanoFunctionGroup(object):
    """docstring for TheanoFunctionGroup"""
    def __init__(self, name, layer, outputs, **theano_kwargs):
        super(TheanoFunctionGroup, self).__init__()
        
        self.name = name
        self.layer = layer
        self.outputs = outputs
        self.theano_kwargs = theano_kwargs

        self.functions = {}
        try :
            streams = self.outputs.streams
        except AttributeError :
            try:
                streams = self.outputs.keys()
            except KeyError:
                raise ValueError("Unable to derive streams form object: '%s'" % self.outputs)

        for stream in streams :
            self.functions[stream] = None

        self.updates = set()
        self.mustInit = True

    def allowUpdates(self, stream) :
        if stream not in self :
            raise ValueError("Output has no stream: '%s'" % stream)
        self.updates.add(stream)

    def removeUpdates(self, stream) :
        if stream not in self :
            raise ValueError("Output has no stream: '%s'" % stream)
        self.updates.remove(stream)
    
    def init(self) :
        if self.mustInit :
            for stream in self.functions :
                update = False
                if stream in self.updates :
                    update = True
                if self.functions[stream] is None :
                    self.functions[stream] = TheanoFunction("%s.%s" %(self.name, stream), self.layer, self.outputs, stream=stream, update = update, **self.theano_kwargs)
            self.mustInit = False

    def __getitem__(self, stream) :
        self.init()
        return self.functions[stream]

    def __setitem__(self, stream, v) :
        if stream not in self :
            raise ValueError("Cannot add a function. Output has no stream: '%s'" % stream)

        self.functions[stream] = v
        self.mustInit = True

    def __contains__(self, stream) :
        """check if the stream is supported"""
        return stream in self.functions
