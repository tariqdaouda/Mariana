from collections import OrderedDict
import theano, sys, numpy
import sys

import Mariana.candies as MCAN
import Mariana.settings as MSET
import Mariana.custom_types as MTYPES

class UpdateStore(object):
    """docstring for UpdateStore"""
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
    """docstring for Updates"""
    def __init__(self, output_layer, loss):
        super(Updates, self).__init__()
        self.output_layers = [output_layer]
        self.layers = [output_layer]
        self.layers.extend(output_layer.dependencies.values())
        self.loss = loss

        self.store = UpdateStore()

    def extend(self, updates) :
        if updates :
            self.output_layers.extend(updates.output_layers)
            self.layers.extend(updates.layers)
            self.loss += updates.loss

    def compile(self) :
        def _apply(sc, store, layer, entity, param_name, cute_name, loss) :
            res = sc.apply(layer, layer, param_name, loss)
            if res is not None :
                gup = res.parameter
                store.add(gup.parameter, gup.update, gup.gradient, cute_name)
                for gup_co in res.coParameters :
                    cute_name2 = "%s.%s" %(cute_name, gup_co.name)
                    store.add(gup_co.parameter, gup_co.update, gup_co.gradient, cute_name2)

        scCheck = set()
        for layer in self.layers :
            for sc in layer.abstractions["scenari"] :
                if sc not in scCheck :
                    for gup_free in sc.freeParameters.parameters :
                        pname = "%s.%s.%s" %(output_layer.name, sc.__class__.__name__, gup_free.name)
                        self.store.add(gup_free.parameter, gup_free.update, gup_free.gradient, pname)
                    scCheck.add(sc)

                for k in layer.getParameterNames() :
                    pname = "%s.%s" %(layer.name, k)
                    _apply(sc, self.store, layer, layer, k, pname, self.loss)

                for abst in layer.abstractions :
                    try:
                        names = abst.getParameterNames()
                    except Exception as e:
                        pass
                    else :
                        for k in names:
                            pname = "%s.%s.%s" %(layer.name, abst.__class__.__name__, k)
                            _apply(sc, self.store, layer, abst, k, pname, self.loss)

class TheanoFunctionHandle(object) :
    """
    This class encapsulates a Theano function.
    TheanoFunction objects should be defined as self attributes in the setCustomTheanoFunctions() function of output layers.
    It will also generate custom error messages whose verbosity depends on Mariana.settings.VERBOSE. Set it to False to get quieter
    error messages.
    """

    def __init__(self, name, layer, output, stream, updates=None, **theano_kwargs) :
        """
        :param Output layer: the output layer the function should be applied to
        :param list updates: boolean defining if updates should be applied
        :param dict \*\*theano_kwargs: additional arguments to passed to the real theano function underneath
        :parama stream: usually something like "train" or "test". Defines if regularizations and decorators should be applied
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
        self.updates = updates
        self.layer = layer
        self.stream = stream
        self.theano_kwargs = theano_kwargs

        self.inputs = _bckTrckInputs(layer, stream)
        for k, v in layer.__dict__.iteritems() :
            if v.__class__ is MTYPES.Targets :
                self.inputs["%s.%s" % (layer.name, k)] = v[stream]

        self.output = output
        self.theano_fct = None

    def hasUpdates(self) :
        return self.updates is not None

    def __add__(self, other) :
        if other.__class__ is TheanoFunction :
            if other.isCompiled() :
                raise TypeError("Added value cannot be an already compiled function")
            other._addHandle(self)
            return other
        elif other.__class__ is not TheanoFunctionHandle :
            raise TypeError("Added value must be another valid function")

        return TheanoFunction([self, other])

    def _develop(self) :
        if not self.theano_fct :
            self.theano_fct = TheanoFunction([self])
            
    def __call__(self, *args, **kwargs) :
        self._develop()
        return self.theano_fct.run(*args, **kwargs)

    def __getattr__(self, k) :
        self._develop()
        if hasattr(self.theano_fct, k) :
            return getattr(self.theano_fct, k)

class TheanoFunction(object) :

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
        return self.theano_fct is not None

    def _addHandle(self, theano_handle) :
        self.theano_handles.append(theano_handle)
    
    def _compile(self):

        if not self.isCompiled() :
            self.inputs = OrderedDict()
            self.outputs = OrderedDict()
            self.updates = None
            
            self.fct_inputs_varToName = OrderedDict()
            self.fct_inputs = OrderedDict()
            
            self.theano_kwargs = {}
            for handle in self.theano_handles :
                self.theano_kwargs.update(handle.theano_kwargs)
                
                for k, v in handle.inputs.iteritems() :
                    self.inputs[k] = v
                    self.fct_inputs_varToName[v] = k
                    self.fct_inputs[k] = None
                
                self.outputs["%s.%s" % (handle.layer.name, handle.name)] = handle.output
                
                if handle.hasUpdates() :
                    if self.updates is None :
                        self.updates = handle.updates
                    else :
                        self.updates.extend(handle.updates)

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

    def run(self, inputs = {}) :
        self._compile()
        self._parseInputs(inputs)

        fres = iter(self.theano_fct(*self.fct_inputs.values()))
        
        for k in self.outputs.iterkeys() :
            self.results[k] = fres.next()

        return self.results

    def getGradients(self, inputs={}) :
        self._compile()
        if not self.updates :
            raise TypeError("Function has no updates, cannot have gradients")

        self._parseInputs(inputs)

        if not self.gradients_fct :
            self.gradients_fct = theano.function(inputs = self.inputs.values(), outputs = self.updates.store.gradients.values(), **self.theano_kwargs)

        fres = iter(self.gradients_fct(*self.fct_inputs.values()))
        results = OrderedDict()
        for k in self.updates.store.names["gradients"].itervalues() :
            results[k] = fres.next()

        return results
        
    def getUpdates(self, inputs={}) :
        self._compile()
        if not self.updates :
            raise TypeError("Function has no updates")
        self._parseInputs(inputs)

        if not self.updates_fct :
            self.updates_fct = theano.function(inputs = self.inputs.values(), outputs = self.updates.store.updates.values(), **self.theano_kwargs)

        fres = iter(self.updates_fct(*self.fct_inputs.values()))
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
        return self.run(*args, **kwargs)

    def __repr__(self) :
        return "<Mariana Theano Fct '%s'>" % self.name

    def __str__(self) :
        return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)
