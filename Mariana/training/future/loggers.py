import numpy

__all__ = ["Logger_ABC", "ParameterValue_ABC", "ParameterMean", "ParameterMin", "ParameterMax"]

class Logger_ABC(object) :

    def log(self, trainer) :
        NotImplemented("Must be implemented in child")

class ParameterValue_ABC(Logger_ABC) :

    def __init__(self, paramList = None) :
        super(ParameterValue_ABC, self).__init__()
        if paramList is not None :
            temp = []
            for lp in paramList :
                slp = lp.split(".")
                if len(slp) == 1 :
                    temp.append( (lp, None ) )
                elif len(slp) == 2 :
                   temp.append( (slp[0], slp[1]) )
                else :
                    raise ValueError("Parameters in parameter list should have on of the following formats: layerName.paramName, or layerName. Got: %s" % lp)

            self.paramList = temp
        else :
            self.paramList = paramList

    def getValue(self, v) :
        NotImplemented("Must be implemented in child")

    def getKey(self, layer, paramName) :
        NotImplemented("Must be implemented in child")
    
    def log(self, trainer) :
        store = []
        if self.paramList is None :
            lnames = trainer.model.layers.keys()

            for lname in lnames :
                layer = trainer.model[lname]
                for k, v in layer.getParameterDict().iteritems() :
                    store.append( (self.getKey(layer, k) , self.getValue(v) ) )
        else :
            for lname, paramName in self.paramList :
                layer = trainer.model[lname]
                if paramName is None :
                    for k, v in layer.getParameterDict().iteritems() :
                        store.append( (self.getKey(layer, k) , self.getValue(v) ) )      
                else :
                    v = getattr(layer, paramName)
                    store.append( (self.getKey(layer, paramName) , self.getValue(v) ) )
                
        return store

class ParameterMean(ParameterValue_ABC) :

    def getKey(self, layer, paramName) :
        return "%s.%s.MEAN"%(layer.name, paramName.upper())
    
    def getValue(self, v) :
        return numpy.mean(v.get_value())

class ParameterStd(ParameterValue_ABC) :

    def getKey(self, layer, paramName) :
        return "%s.%s.STD"%(layer.name, paramName.upper())
    
    def getValue(self, v) :
        return numpy.std(v.get_value())

class ParameterMin(ParameterValue_ABC) :

    def getKey(self, layer, paramName) :
        return "%s.%s.MIN"%(layer.name, paramName.upper())
    
    def getValue(self, v) :
        return numpy.min(v.get_value())

class ParameterMax(ParameterValue_ABC) :

    def getKey(self, layer, paramName) :
        return "%s.%s.MAX"%(layer.name, paramName.upper())
    
    def getValue(self, v) :
        return numpy.max(v.get_value())

class AbstractionHyperParameters(Logger_ABC) :

    def __init__(self, layerNames = None) :
        self.layerNames = layerNames
        self.abstractions = ["initializations", "activation", "decorators", "regularizationObjects", "learningScenario", "costObject"]
        self.store = []

    def updateStore(self, layer, abstraction, absType) :
        if abstraction is None :
            key = "%s.%s" % (layer.name, absType)
            self.store.append( (key, None) )
            return
        
        try :
            abName = abstraction.name
        except :
            abName = abstraction.__class__.__name__

        if hasattr(abstraction, "hyperParameters") :
            for hp in abstraction.hyperParameters :
                value = getattr(abstraction, hp)
                key = "%s.%s.%s" % (layer.name, abName, hp)
                self.store.append( (key, value) )
        else :
            key = "%s.%s" % (layer.name, abName)
            self.store.append( (key, True) )

    def log(self, trainer) :
        if len(self.store) > 0 :
            return self.store
        
        if self.layerNames is not None :
            layerNames = self.layerNames
        else :
            layerNames = iter(trainer.model.layers)

        for lname in layerNames :
            layer = trainer.model[lname]
            for absType in self.abstractions :
                try :
                    iterAbs = iter(getattr(layer, absType))
                except TypeError:
                    print layer, absType
                    self.updateStore(layer, getattr(layer, absType), absType)
                except AttributeError :
                    pass
                else :
                    for abstraction in iterAbs :
                        self.updateStore(layer, abstraction, absType)
        
        return self.store

class Scores(Logger_ABC) :

    def __init__(self, summarize=True) :
        self.store = {}
        self.maxValues = {}
        self.minValues = {}
        self.summarize = summarize

    def log(self, trainer) :
        store = []
        for setName, sets in trainer.store["scores"].iteritems() :
            for outputName, outputs in sets.iteritems() :
                for fctName, value in outputs.iteritems() :
                    key = "%s.%s.%s" % (setName, outputName, fctName)
                    isMax = False
                    isMin = False
                    try :
                        if self.maxValues[key][1] < value :
                            self.maxValues[key] = (trainer.store["runInfos"]["epoch"], value)
                            isMax = True
                    except KeyError :
                        self.maxValues[key] = (trainer.store["runInfos"]["epoch"], value)
                        isMax = True

                    try :
                        if self.minValues[key][1] > value :
                            self.minValues[key] = (trainer.store["runInfos"]["epoch"], value)
                            isMin = True
                    except KeyError :
                        self.minValues[key] = (trainer.store["runInfos"]["epoch"], value)
                        isMin = True
                    
                    if self.summarize :     
                        if isMax :
                            hndlMax = "+MAX+"
                        else :
                            hndlMax = "%.3f@%d" % (self.maxValues[key][1], self.maxValues[key][0])

                        if isMin :
                            hndlMin = "-MIN-"
                        else :
                            hndlMin = "%.3f@%d" % (self.minValues[key][1], self.minValues[key][0])

                        strV = "%.3f@%d [%s, %s], " % (value, trainer.store["runInfos"]["epoch"], hndlMin, hndlMax )
                        store.append( ( key, strV) )          
                    else :
                        store.append( ( key, value) )
                        store.append( ("%s.MAX.EPOCH" % key, self.maxValues[key][0]) )
                        store.append( ("%s.MAX" % key, self.maxValues[key][1]) )
                        store.append( ("%s.IS.MAX" % key, isMax) )

                        store.append( ("%s.MIN.EPOCH" % key, self.minValues[key][0]) )
                        store.append( ("%s.MIN" % key, self.minValues[key][1]) )
                        store.append( ("%s.IS.MIN" % key, isMin) )

        return store