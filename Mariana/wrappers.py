from collections import OrderedDict
import theano
import theano.tensor as tt

import Mariana.candies as MCAN
import Mariana.settings as MSET
import Mariana.scenari as MS
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
        
        #append outputs optimization rules at the top level
        self.loss = 0
        optimizers = {}
        names = {}
        for o in self.output_layers :
            self.loss += o.loss[self.stream]
            optimizers[o] = o.abstractions["learningScenari"]
            names[o] = o.name
            inheritables = []
            for ls in o.abstractions["learningScenari"] :
                if ls.isInheritable() :
                    inheritables.append(ls)

            for abstraction in o.getTrainableAbstractions() :
                optimizers[abstraction] = list(inheritables)
                names[abstraction] = "%s.%s" % (o.name. abstraction.name)
        
            for l in o.dependencies.itervalues() :
                if l not in optimizers :
                    names[l] = l.name
                    optimizers[l] = []
                if len(l.abstractions["learningScenari"]) > 0 :
                    if not isinstance(l.abstractions["learningScenari"][0], MS.Independent) :
                        optimizers[l].extend(inheritables)
                else :
                    optimizers[l].extend(inheritables)

                for abstraction in l.getTrainableAbstractions() :
                    if abstraction not in optimizers :
                        names[abstraction] = "%s.%s" % (l.name. abstraction.name)
                        optimizers[abstraction] = []
                    
                    if len(abstraction.abstractions["learningScenari"]) > 0 :
                        if not isinstance(abstraction.abstractions["learningScenari"][0], MS.Independent) :
                            optimizers[abstraction].extend(inheritables)
                    else :
                        optimizers[abstraction].extend(inheritables)

        #append specific optimization rules
        for abstraction in optimizers :
            if abstraction not in self.output_layers :
                optimizers[abstraction].extend(abstraction.abstractions["learningScenari"])
                for reg in abstraction.abstractions["regularizations"] :
                    self.loss = reg.apply(abstraction, self.loss)
        
        preStore = {}
        appliedOptim = {}
        for abstraction, scenari in optimizers.iteritems() :
            appliedOptim[abstraction] = set()
            for sc in scenari :
                if sc not in appliedOptim[abstraction] :
                    appliedOptim[abstraction].add(sc)
                    for paramName in abstraction.getParameters() :
                        previousOptim=None
                        optim = sc.apply(abstraction=abstraction, parameterName=paramName, loss=self.loss, previous=previousOptim)
                        name = "%s.%s" % (names[abstraction], paramName),
                        if optim :
                            preStore[name] = {
                                "parameter": optim.parameter,
                                "gradient": optim.gradient,
                                "update" : optim.update,
                                "name" : name,
                                "coParameters": []
                            }
                            for coParam in optim.coParameters :
                                preStore[name]["coParameters"].append(
                                    {
                                        "parameter": coParam.parameter,
                                        "gradient": coParam.gradient,
                                        "update" : coParam.update,
                                        "name" : "%s.%s.%s" % (name, coParam.name),
                                    }
                                )
                            previousOptim = optim
        
        for gup in preStore.itervalues() :
            self.store.add( gup["parameter"], gup["gradient"], gup["update"], gup["name"] )
            for coParam in gup["coParameters"] :
                self.store.add( coParam["parameter"], coParam["gradient"], coParam["update"], coParam["name"] )

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
                if isinstance(v, MTYPES.Inputs) and not v.isTied(stream):
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
            if isinstance(v, MTYPES.Targets) and not v.isTied(stream) :
                self.inputs["%s.%s" % (layer.name, k)] = v[stream]

        try :
            self.output = output[stream]
        except :
            raise ValueError("Output does not have a stream: '%s'" % stream)

        # self.theano_fct = None

    def hasUpdates(self) :
        """returns True if the function updates the parameters, False other wise."""
        return self.update
    
class KolokoTheanoFunction(object) :
    """
    This class encapsulates a just-in-time compiled Theano function.
    """
    def __init__(self) :
        super(KolokoTheanoFunction, self).__init__()

        self.theano_handles = []
        self.theano_fct = None
        self.gradients_fct = None
        self.updates_fct = None

        self.inputs = None
        self.outputs = None
        self.updates = None
    
        self.perfomUpdates = None
        self.stream = None
            
    def isCompiled(self) :
        """Has the compildation already happend?"""
        return self.theano_fct is not None

    def __add__(self, mar_theano_fct) :
        if self.stream != mar_theano_fct.stream :
            raise ValueError("All functions must be on the same stream. Got: '%s' and '%s' " % (self.stream, mar_theano_fct.stream))

        fct = KolokoTheanoFunction()
        fct.stream = self.stream

        for hand in self.theano_handles :
            fct.addHandle(hand)
    
        for hand in mar_theano_fct.theano_handles :
            fct.addHandle(hand)

        return fct

    def addHandle(self, theano_handle) :
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
        # if len(inputs) != len(self.inputs_varToName) :
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

        fct_inputs = OrderedDict()
        for param, pname in self.inputs_varToName.iteritems() :
            # print param, pname, inputs
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
        if not self.isCompiled() :
            if len(self.theano_handles) == 1 :
                return "< Uncompiled Mariana Theano Fct: %s.%s >" %(self.theano_handles[0].layer.name, self.theano_handles[0].name)
            else :
                s =[]
                for hand in self.theano_handles :
                    s.append("%s.%s" %(hand.layer.name, hand.name))
                return "< Uncompiled Mariana Theano Fct Mix: %s >" % (' + '.join(s) )
        else :
            args = [] 
            for k, v in self.inputs_varToName.items() :
                args.append("%s: %s" %(v, k))

            return "<Compiled Mariana Theano Fct '%s'. Arguments: '%s'>" % (id(self), ', '.join(args))

class TheanoFunction(KolokoTheanoFunction):
    """docstring for TheanoFunction"""
    def __init__(self, name, layer, output, stream, update=False, **theano_kwargs):
        super(TheanoFunction, self).__init__()
        self.stream = stream
        self.theano_handles = [TheanoFunctionHandle(name, layer, output, stream, update=update, **theano_kwargs)]

class TheanoFunctionGroup(object):
    """High level that wraps a group of function (one for every stream)"""
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
        """Apply updates on a given stream"""
        if stream not in self :
            raise ValueError("Output has no stream: '%s'" % stream)
        self.updates.add(stream)

    def removeUpdates(self, stream) :
        """Removes updates from a given stream"""
        if stream not in self :
            raise ValueError("Output has no stream: '%s'" % stream)
        self.updates.remove(stream)
    
    def init(self) :
        """Initialize the group"""
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
