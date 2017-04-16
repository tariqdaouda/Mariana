from collections import OrderedDict
import theano, sys, numpy
import sys

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
    
    def add(self, parameter, update, gradient, name) :
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

    def merge(self, updates) :
        """Merge tow updates by combining their outputs and adding theirs losses"""
        if not isinstance(updates, self.__class__) :
            raise ValueError("Parameter must be an instance of '%s" % self.__class__.__name__)
        if updates :
            self.output_layers.extend(updates.output_layers)
            self.loss += updates.loss

    def compile(self) :
        """Derive the updates and gradients for every parameter in the network"""
        def _apply(sc, store, layer, entity, param_name, cute_name, loss) :
            try :
                previous = store[cute_name]
            except KeyError :
                previous = None

            store[cute_name] = sc.apply(layer, entity, param_name, loss, previous)
            
        self.loss = 0
        optimizers = {}
        for o in self.output_layers :
            self.loss += o.loss[self.stream]
            optimizers[o] = o.abstractions["learningScenari"]
            for l in o.dependencies.itervalues() :
                for sc in o.abstractions["learningScenari"] :
                    if sc.inheritable :
                        try :
                            optimizers[l].append(sc)
                        except KeyError :
                            optimizers[l] = [sc]
                
        for o in self.output_layers :
            for l in o.dependencies.itervalues() :
                try :
                    optimizers[l].extend(l.abstractions["learningScenari"])
                except KeyError :
                    if len(l.abstractions["learningScenari"]) == 0 and len(l.parameters) > 0 :
                        raise ValueError("Layer: '%s' has trainable parameters but no defined learning scenario. If you don't want to train it, give it the Fixed() scenario." % l.name)
                    else :
                        optimizers[l] = l.abstractions["learningScenari"]

                for reg in l.abstractions["regularizations"] :
                    self.loss = reg.apply(l, self.loss)

        scCheck = set()
        tmpStore = {}
        for layer, scenari in optimizers.iteritems() :
            for sc in scenari :
                if sc not in scCheck :
                    for gup_free in sc.freeParameters.parameters :
                        pname = "%s.%s.%s" %(output_layer.name, sc.__class__.__name__, gup_free.name)
                        self.store.add(gup_free.parameter, gup_free.update, gup_free.gradient, pname)
                    scCheck.add(sc)

                for k in layer.parameters.iterkeys() :
                    pname = "%s.%s" %(layer.name, k)
                    _apply(sc, tmpStore, layer, layer, k, pname, self.loss)

                for abst in layer.abstractions.itervalues() :
                    if isinstance(abst, list) :
                        for abstt in abst :
                            for k in abstt.parameters.iterkeys() :
                                pname = "%s.%s.%s" %(layer.name, abst.__class__.__name__, k)
                                _apply(sc, tmpStore, layer, abst, k, pname, self.loss)
                    else :
                        for k in abst.parameters.iterkeys() :
                            pname = "%s.%s.%s" %(layer.name, abst.__class__.__name__, k)
                            _apply(sc, tmpStore, layer, abst, k, pname, self.loss)
        
        for cute_name, res in tmpStore.iteritems() :
            if res is not None :
                gup = res.parameter
                self.store.add(gup.parameter, gup.update, gup.gradient, cute_name)
                for gup_co in res.coParameters :
                    cute_name2 = "%s.%s" %(cute_name, gup_co.name)
                    self.store.add(gup_co.parameter, gup_co.update, gup_co.gradient, cute_name)

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

        self.output = output[stream]
        self.theano_fct = None

    def hasUpdates(self) :
        """returns True if the function updates the parameters, False other wise."""
        return self.update

    def __add__(self, other) :
        """Add to handle together using the '+' operator. Losses will be added up and updates will be merged"""
        if self.stream != other.stream :
            raise TypeError("All functions must be in the same stream %s != %s" %(self.stream, other.stream))
        
        if other.__class__ is TheanoFunction :
            if other.isCompiled() :
                raise TypeError("Cannot add an already compiled function")
            other._addHandle(self)
            return other
        elif other.__class__ is not TheanoFunctionHandle :
            raise TypeError("Added value must be another valid function")

        return TheanoFunction([self, other])

    def _develop(self) :
        """Compile the inner theano function"""
        if not self.theano_fct :
            self.theano_fct = TheanoFunction([self])

    def __call__(self, *args, **kwargs) :
        """Call the inner theano functiom"""
        self._develop()
        return self.theano_fct.run(*args, **kwargs)

    def __getattr__(self, k) :
        """return the theano function attributes"""
        self._develop()
        if hasattr(self.theano_fct, k) :
            return getattr(self.theano_fct, k)

class TheanoFunction(object) :
    """
    This class encapsulates a just-in-time compiled Theano function.
    """
    def __init__(self, theano_handles) :
        super(TheanoFunction, self).__init__()

        self.theano_handles = theano_handles
        self.theano_fct = None
        self.gradients_fct = None
        self.updates_fct = None

        self.inputs = None
        self.outputs = None
        self.updates = None
            
    def isCompiled(self) :
        """Has the compildation already happend?"""
        return self.theano_fct is not None

    def _addHandle(self, theano_handle) :
        """Add a handle to the current definition. Losses will be added up and updates will be merged"""
        self.theano_handles.append(theano_handle)
    
    def _compile(self):
        """Compile the function just-in-time according the definitions given by the handles."""
        if not self.isCompiled() :
            self.inputs = OrderedDict()
            self.outputs = OrderedDict()
            self.updates = None
            
            self.fct_inputs_varToName = OrderedDict()
            
            self.theano_kwargs = {}
            for handle in self.theano_handles :
                self.theano_kwargs.update(handle.theano_kwargs)
                
                for k, v in handle.inputs.iteritems() :
                    self.inputs[k] = v
                    self.fct_inputs_varToName[v] = k
                    
                self.outputs["%s.%s" % (handle.layer.name, handle.name)] = handle.output
                
                if handle.hasUpdates() :
                    if self.updates is None :
                        self.updates = Updates(handle.layer, handle.stream)
                    else :
                        self.updates.merge(Updates(handle.layer, handle.stream))

            if self.updates :
                self.updates.compile()
                updates = self.updates.store.updates
            else :
                updates = {}
            
            self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.outputs.values(), updates = updates, **self.theano_kwargs)

            if MSET.DEVICE_IS_GPU :
                if str(self.getToposort()).find("float64") > -1:
                    msg = "There are some float64s that do not fit on the GPU and will slow down the computations.\nPlease consider:"
                    msg += "\n\t* Launching with THEANO_FLAGS=device=gpu,floatX=float32 python <your script>.py."
                    msg += "\n\t* If you have any dmatrix, dvector or dscalar in your code replace them with matrix, vector, scalar."
                    MCAN.friendly("Run device", msg, warning = True)
            
            self.results = OrderedDict()

    def _parseInputs(self, inputs = {}) :
        """parse the inputs to the function and tells you if there's anything missing, in a friendly and human readable way"""
        nbArgs = 0
        fct_inputs = OrderedDict()
        for param, pname in self.fct_inputs_varToName.iteritems() :
            try :
                fct_inputs[param] = inputs[pname]
            except KeyError :
                try:
                    fct_inputs[param] = inputs[param]
                except KeyError as e:
                    raise TypeError("missing argument '%s' " % pname)
            nbArgs += 1

        if nbArgs != len(self.fct_inputs_varToName) :
            args = str(fct_inputs.keys()).replace("[", "").replace("]", "")
            raise TypeError("Function expects the following arguments: %s, %d provided" % (args, nbArgs) )

        return fct_inputs

    def run(self, inputs = {}) :
        """run the function and return the results"""
        self._compile()
        fct_inputs = self._parseInputs(inputs)

        fres = iter(self.theano_fct(*fct_inputs.values()))
        
        for k in self.outputs.iterkeys() :
            self.results[k] = fres.next()

        return self.results

    def getGradients(self, inputs={}) :
        """return the gradients that would be performed"""
        self._compile()
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
        self._compile()
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
        return "<Mariana Theano Fct '%s'>" % self.name

    def __str__(self) :
        return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)
