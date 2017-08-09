import Mariana.layers as ML


# pkl = {train: {3:, 5:}, test...}
# dataset = MD.ClassDataset(pkl)
# d = DataFeeder(fct)
# d.streamTranslate("10%", "train")
# d.feed("input.inputs", data.images)
# d.sample(100)
# d.all()

class DatasetMapper(object):
    """a DatasetMapper maps Input and Output layer to the data they must receive.
    It's much less complicated than it sounds cf. doc for map(), you can ignore the others.
    The DatasetMapper is intended to be used with a Trainer

    :param miniBatchSize: Size of miniBatches, use None for the complete dataset
    :param rerollFreq:How many epochs before internally shuffling the subsets). Freq=10 means 1 reroll each 10 epochs. Use a None to prevent rerolls

    """

    def __init__(self, runFunctionName, miniBatchSize=None, rerollFreq=1):
        self.datasets = set()
        self.maps = {}
        self.layersByName = {}
        self.inputLayers = []
        self.outputLayers = []
        self.minLength = float('inf')
        self.minFullLength = float('inf')

        self.miniBatchSize = miniBatchSize
        if miniBatchSize is None :
            self.rerollFreq = None
        else :
            self.rerollFreq = rerollFreq

        self.epochNumber = 0
        self.batchNumber = 0
        self.runFunctionName = runFunctionName
        self.runFunction = None

    def mapInput(self, layer, setHandle) :
        """
        Maps an input or an output layer to Dataset's subset::
            
            i = InputLayer(...)

            trainSet = RandomSeries(images = train_set[0], classes = train_set[1])
            
            DatasetMapper = dm
            dm.map(i, trainSet.images)
        """
        import Mariana.settings as MSET

        if MSET.TYPE_INPUT_LAYER not in layer.types :
            raise ValueError("%s is not an input layer (type: %s)" % (layer.name, layer.types))

        self.inputLayers.append(layer)
        self.maps[layer] = ( (layer.name, setHandle), )
        self.layersByName[layer.name] = layer
        
        self.datasets.add(setHandle.dataset)
        if len(setHandle.dataset) < self.minLength : 
            self.minLength = len(setHandle.dataset) 
        if setHandle.dataset.getFullLength() < self.minFullLength : 
            self.minFullLength = setHandle.dataset.getFullLength()

        if self.runFunction is None :
            self.runFunction = getattr(layer.network, self.runFunctionName)

    def mapOutput(self, layer, setHandle, inputName = 'targets') :
        """
        Maps an input or an output layer to Dataset's subset::
            
            o = OutputLayer(...)

            trainSet = RandomSeries(images = train_set[0], classes = train_set[1])
            
            DatasetMapper = dm
            dm.map(o, trainSet.classes, "targets")
        
        inputName: train and test functions often need addictional inputs such as the targets. With this
        parameter you can associate one of these additional inputs (as defined in a theano function) to
        a dataset. This argument is optional, the default value is 'targets' which should work for all 
        out of the box Mariana stuff. 
        """
        import Mariana.settings as MSET
        
        if MSET.TYPE_OUTPUT_LAYER not in layer.types :
            raise ValueError("%s is not an output layer (type: %s)" % (layer.name, layer.types))

        self.outputLayers.append(layer)
        self.layersByName[layer.name] = layer
        
        k = (inputName, setHandle)

        try :
            self.maps[layer].append(k)
        except KeyError :
            self.maps[layer] = [ k ]
        
        self.datasets.add(setHandle.dataset)
        if len(setHandle.dataset) < self.minLength : 
            self.minLength = len(setHandle.dataset)
        if setHandle.dataset.getFullLength() < self.minFullLength : 
            self.minFullLength = setHandle.dataset.getFullLength()

        if self.runFunction is None :
            self.runFunction = getattr(layer.network, self.runFunctionName)
            
    def reroll(self, force=False) :
        """internally shuffle subsets periodically according to reroll frequency. This is called once by the trainer at each epoch"""
        if force or ( self.rerollFreq is not None and self.epochNumber%self.rerollFreq == 0 ):
            for d in self.datasets :
                d.reroll()

    def next(self, layerList=None, strict=False) :
        """returns the next batch of data
        
        :param list layerList: The list of layers for which to return data. If None, will return for all layers.
        :param bool strict: Will raise a KeyError exception if a layer from the list has no associated map, default behaviour is to ignore it.

        """
        if self.miniBatchSize is None :
            if self.batchNumber > 0 :
                self.batchNumber = 0
                self.epochNumber += 1
                self.reroll()
                raise StopIteration("That was the last batch")

            batch = self.getAll(layerList, strict)
        else :
            try :
                batch = self.getBatch(self.batchNumber*self.miniBatchSize, self.miniBatchSize, layerList, strict)
            except IndexError :
                self.batchNumber = 0
                self.epochNumber += 1
                self.reroll()
                raise StopIteration("That was the last batch")
                
        self.batchNumber += 1
        return batch

    def getBatch(self, i, size, layerList=None, strict=False) :
        """
        Returns a dictionary::
            
                layer.name => elements

        :param list layerList: The list of layers for which to return data. If None, will return for all layers.
        :param bool strict: Will raise a KeyError exception if a layer from the list has no associated map, default behaviour is to ignore it.
        """
        if i >= self.minLength :
            raise IndexError("index i '%s', out of range '%s'" % (i, size))

        res = {}
        if layerList is None :
            layers = self.maps.keys()
        else :
            layers = layerList

        for l in layers :
            try :
                layer = self.layersByName[l]
            except KeyError :
                layer = l
            try :
                for name, handle in self.maps[layer] :
                    res[name] = handle.dataset.get(handle.subset, i, size)
            except KeyError:
                if strict :
                    raise KeyError("There's no layer: %s" % layer)
        return res

    def getAll(self, layerList=None, strict=False) :
        """Same as getBatch() but returns the totality of the subset for each layer

        :param list layerList: The list of layers for which to return data. If None, will return for all layers.
        :param bool strict: Will raise a KeyError exception if a layer from the list has no associated map, default behaviour is to ignore it.
        """

        if layerList is None :
            layers = self.maps.keys()
        else :
            layers = layerList

        res = {}
        for l in layers :
            try :
                layer = self.layersByName[l]
            except KeyError :
                layer = l
            try :
                for name, handle in self.maps[layer] :
                    res[name] = handle.dataset.getAll(handle.subset)
            except KeyError:
                if strict :
                    raise KeyError("There's no layer: %s" % layer)
        return res

    def getMinFullLength(self) :
        return self.minFullLength

    def __len__(self) :
        return self.minLength

    def __iter__(self):
        return self

