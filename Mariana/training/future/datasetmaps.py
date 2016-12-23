import Mariana.layers as ML
import numpy, random, theano
from collections import OrderedDict

__all__ = ["DatasetHandle", "Dataset_ABC", "Series", "RandomSeries", "ClassSets", "DatasetMapper"]

class DatasetHandle(object) :
    """Handles represent a couple (dataset, subset). They are returned by
    Datasets and are mainly intended to be used as arguments to a DatasetMapper
    to associate layers to data."""

    def __init__(self, dataset, subset) :
        self.dataset = dataset
        self.subset = subset

    def __repr__(self) :
        return "<Handle %s.%s>" % (self.dataset.__class__.__name__, self.subset)

class Dataset_ABC(object) :
    """An abstract class representing a datatset.
    A Dataset typically contains and synchronizes and manages several subsets.
    Datasets are mainly intended to be used with DatasetMappers.
    It should take subsets as constructor aguments and return handles through
    the same interface as attributes (with a '.subset').
    """

    def get(self, subset, i, n) :
        """Returns n element from a subset, starting from position i"""
        raise NotImplemented("Should be implemented in child")

    def getAll(self, subset) :
        """return all the elements from a subset"""
        raise NotImplemented("Should be implemented in child")

    def reroll(self) :
        """re-initiliases all subset, this is where shuffles are implemented"""
        raise NotImplemented("Should be implemented in child")

    def getHandle(self, subset) :
        """returns a DatasetHandle(self, subset)"""
        raise NotImplemented("Should be implemented in child")

    def getFullLength(self) :
        """returns the total number of examples"""
        raise NotImplemented("Should be implemented in child")

    def __len__(self) :
        raise NotImplemented("Should be implemented in child")

    def __getattr__(self, k) :
        """If no attribute by the name of k is found, look for a
        subset by that name, and if found return a handle"""

        g = object.__getattribute__(self, "getHandle")
        res = g(k)
        if res :
            return res
        raise AttributeError("No attribute or dataset by the name of '%s'" % k)

class Series(Dataset_ABC) :
    """Synchronizes and returns the elements of several lists sequentially.
    All lists must have the same length. It expects arguments in the following form::
            
            trainSet = Series(images = train_set[0], classes = train_set[1])"""
    
    def __init__(self, **kwargs) :

        self.lists = {}
        self.length = 0
        for k, v in kwargs.iteritems() :
            if self.length == 0 :
                self.length = len(v)
            else :
                if len(v) != self.length :
                    raise ValueError("All lists must have the same length, previous list had a length of '%s', list '%s' has a length of '%s" % (self.length, k, len(v)))
            self.lists[k] = numpy.asarray(v)

    def reroll(self) :
        """Does nothing"""
        pass

    def get(self, subset, i, n) :
        """Returns n element from a subset, starting from position i"""
        return self.lists[subset][i:i + n]

    def getAll(self, subset) :
        """return all the elements from a subset"""
        return self.lists[subset]

    def getHandle(self, subset) :
        """returns a DatasetHandle(self, subset)"""
        if subset in self.lists :
            return DatasetHandle(self, subset)
        return None

    def getFullLength(self) :
        """returns the total number of examples"""
        return len(self)

    def __len__(self) :
        return self.length

    def __repr__(self) :
        return "<%s len: %s, nbLists: %s>" % (self.__class__.__name__, self.length, len(self.lists))

class RandomSeries(Series) :
    """This is very much like Series, but examples are randomly sampled with replacement"""
    def __init__(self, **kwargs) :
        """
        Expects arguments in the following form::
        
                trainSet = RandomSeries(images = train_set[0], classes = train_set[1])"""

        Series.__init__(self, **kwargs)
        self.storeLists = {}
        for k, v in self.lists.iteritems() :
            self.storeLists[k] = numpy.asarray(v)
        
        self.reroll()

    def reroll(self) :
        """shuffles subsets but keeps them synched"""
        indexes = numpy.random.randint(0, self.length, self.length)
        for k in self.lists :
            self.lists[k] = self.storeLists[k][indexes]
            
class ClassSets(Dataset_ABC) :
    """Pass it one set for each class and it will take care of randomly sampling them with replacement.
    All class sets must have at least two elements. If classes do not have the same number of elements
    it will take care of the oversampling for you.
    
    The subsets exposed by a ClassSets are:
        
        * .input, for the raw inputs
        * .classNumber, each class is represent by an int
        * .onehot, onehot representation of classes

        """

    def __init__(self, sets) :
        """
        Expects arguments in the following form::
        
                trainSet = ClassSets( sets = [ ('cars', train_set[0]), ('bikes', train_set[1]) ] )
        """

        self.classSets = OrderedDict()
        
        self.inputSize = 0
        self.inputShape = None
        for k, v in sets :
            self.classSets[k] = numpy.asarray(v, dtype=theano.config.floatX)
            if len(self.classSets[k].shape) < 2 :
                self.classSets[k] = self.classSets[k].reshape(self.classSets[k].shape[0], 1)

            try :
                inputSize = len(v[0])
            except :
                inputSize = 1

            if self.inputShape is None :
                self.inputSize = inputSize
                self.inputShape = self.classSets[k][0].shape
            elif self.inputSize != inputSize :
                raise ValueError("All class elements must have the same size. Got '%s', while the previous value was '%s'" % ( self.classSets[k][0].shape, self.inputShape ))
                
        self.classNumbers = {}
        self.onehots = {}
        self.nbElements = 0
        i = 0
        for k, v in self.classSets.iteritems() :
            self.nbElements += len(v)
            self.classNumbers[k] = i
            self.onehots[k] = numpy.zeros(len(self.classSets))
            self.onehots[k][i] = 1
            i += 1

        self.samplingStrategy = None
        self.subsets = {
            "input" : None,
            "classNumber" : None,
            "onehot" : None
        }

    def setSampling(self, strategy, arg) :
        """Set the sampling strategy::
        
            * strategy='all_random', arg = k (int). Full data will be made of a random sampling(with replacement) of k elements for each class
            * strategy='filling', arg = class_name (str). The totality of the elements of every class that has as much elements as card(class_name) will be always present. Elements from other classes will be integrated using a random samplings of card(n) elements with replacements.
        """
        if strategy == "all_random" :
            assert arg is not None
            self.sampleSize = arg
        elif strategy == "filling" :
            self.sampleSize = len(self.classSets[arg])
        else:
            raise ValueError("Unkown sampling strategy")

        subsetSize = self.sampleSize * len(self.classSets)
        shape = [subsetSize]
        for v in self.inputShape :
            shape.append(v)
        shape = tuple(shape)

        self.subsets = {
            "input" : numpy.zeros(shape, dtype = theano.config.floatX),
            "classNumber" : numpy.zeros(subsetSize, dtype = theano.config.floatX),
            "onehot" : numpy.zeros( (subsetSize, len(self.classSets)), dtype = theano.config.floatX)
        }
    
        self.samplingStrategy = strategy

    def reroll(self) :
        """internally shuffle subsets"""
        if self.samplingStrategy is None :
            AttributeError("Please use setSampling() to setup a sampling strategy first")

        start = 0
        for k, v in self.classSets.iteritems() :
            end = start+self.sampleSize
            if self.samplingStrategy == "filling" and len(v) == self.sampleSize :
                self.subsets["input"][start : end] = v
            else :
                subIndexes = numpy.random.randint(0, len(v), self.sampleSize)
                #print subIndexes
                self.subsets["input"][start : end] = v[subIndexes]

            self.subsets["classNumber"][start : end] = self.classNumbers[k]
            self.subsets["onehot"][start : end] = self.onehots[k]
            start = end

        size = len(self.subsets["input"])
        indexes = random.sample(xrange(size), size)
        for k in self.subsets :
            self.subsets[k] = self.subsets[k][indexes]

    def get(self, subset, i, size) :
        """Returns n element from a subset, starting from position i"""
        return self.subsets[subset][i:i+size]

    def getAll(self, subset) :
        """return all the elements from a subset. Oversampled of course if necessary"""
        return self.subsets[subset]

    def getHandle(self, subset) :
        """returns a DatasetHandle(self, subset)"""
        if subset in self.subsets :
            return DatasetHandle(self, subset)

    def getFullLength(self) :
        """returns the total number of examples"""
        return self.nbElements

    def __len__(self) :
        return self.sampleSize * len(self.classSets)

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
        import Mariana.network as MNET

        if layer.type != MNET.TYPE_INPUT_LAYER :
            raise ValueError("%s is not an input layer (type: %s)" % (layer.name, layer.type))

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
        import Mariana.network as MNET
        
        if layer.type != MNET.TYPE_OUTPUT_LAYER :
            raise ValueError("%s is not an output layer (type: %s)" % (layer.name, layer.type))

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

