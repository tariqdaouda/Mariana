import cPickle, time, sys, os, traceback, types, json, signal
import numpy
from collections import OrderedDict
import theano.tensor as tt

import Mariana.network as MNET
import Mariana.training.future.recorders as MREC
import Mariana.training.future.stopcriteria as MSTOP
import Mariana.candies as MCAN
import Mariana.settings as MSET

__all__ = ["Trainer_ABC", "DefaultTrainer"]

class OutputTrainingSchedule_ABC(object) :

    def __init__(self, name) :
        self.name = self.__class__.__name__

    def run(aMap) :
        NotImplemented("Must be implemented in child")

class Synchronous(OutputTrainingSchedule_ABC) :

    def __init__(self, outputLayers=None) :
        self.outputLayers = outputLayers

    def run(self, aMap) :
        layerList = aMap.inputLayers
        if self.outputLayers :
            outputs = self.outputLayers
        else :
            outputs = aMap.outputLayers

        scores = {}
        for batchData in aMap :
            for output in outputs :
                layerList.append(output)
                res = aMap.runFunction(output, **batchData)
                scores[output.name] = {}
                for k, v in res.iteritems() :
                    try :
                        scores[output.name][k].append(v)
                    except KeyError:
                        scores[output.name][k] = [v]
                layerList.pop(-1)
                
        for output in aMap.outputLayers :
            for k, v in scores[output.name].iteritems() :
                scores[output.name][k] = numpy.mean(v)

        return scores

class Trainer_ABC(object) :
    """This is the general interface of trainer"""

    def __init__(self) :
        """
        The store is initialised to::

        self.store = {
            "runInfos" : {
                "epoch" : 0,
                "runtime" : 0
            },
            "scores" : {}
        }

        """

        self.store = {
            "runInfos" : {
                "epoch" : 0,
                "runtime" : 0
            },
            "scores" : {}
        }

    def start(self, runName, saveIfMurdered=False, **kwargs) :
        """Starts the training and encapsulates it into a safe environement.
        If the training stops because of an Exception or SIGTEM, the trainer
        will save logs, the store, and the last version of the model.
        """

        import simplejson, signal, cPickle

        def _handler_sig_term(sig, frame) :
            _dieGracefully("SIGTERM", None)
            sys.exit(sig)

        def _dieGracefully(exType, tb = None) :
            if type(exType) is types.StringType :
                exName = exType
            else :
                exName = exType.__name__

            death_time = time.ctime().replace(' ', '_')
            filename = "dx-xb_" + runName + "_death_by_" + exName + "_" + death_time
            sys.stderr.write("\n===\nDying gracefully from %s, and saving myself to:\n...%s\n===\n" % (exName, filename))
            self.model.save(filename)
            f = open(filename +  ".traceback.log", 'w')
            f.write("Mariana training Interruption\n=============================\n")
            f.write("\nDetails\n-------\n")
            f.write("Name: %s\n" % runName)
            f.write("pid: %s\n" % os.getpid())
            f.write("Killed by: %s\n" % str(exType))
            f.write("Time of death: %s\n" % death_time)
            f.write("Model saved to: %s\n" % filename)

            if tb is not None :
                f.write("\nTraceback\n---------\n")
                f.write(str(traceback.extract_tb(tb)).replace("), (", "),\n(").replace("[(","[\n(").replace(")]",")\n]"))
            f.flush()
            f.close()
            f = open(filename + ".store.pkl", "wb")
            cPickle.dump(self.store, f)
            f.close()

        signal.signal(signal.SIGTERM, _handler_sig_term)
        if MSET.VERBOSE :
            print "\n" + "Training starts."     
        MCAN.friendly("Process id", "The pid of this run is: %d" % os.getpid())

        try :
            return self.run(runName, **kwargs)
        except MSTOP.EndOfTraining as e :
            print e.message
            death_time = time.ctime().replace(' ', '_')
            filename = "finished_" + runName +  "_" + death_time
            f = open(filename +  ".stopreason.txt", 'w')
            f.write("Name: %s\n" % runName)
            f.write("pid: %s\n" % os.getpid())
            f.write("Time of death: %s\n" % death_time)
            f.write("Epoch of death: %s\n" % self.store["runInfos"]["epoch"])
            f.write("Stopped by: %s\n" % e.stopCriterion.name)
            f.write("Reason: %s\n" % e.message)

            f.flush()
            f.close()
            model.save(filename)
            f = open(filename + ".store.pkl", "wb")
            cPickle.dump(self.store, f)
            f.close()

        except KeyboardInterrupt :
            if not saveIfMurdered :
                raise
            exType, ex, tb = sys.exc_info()
            _dieGracefully(exType, tb)
            raise
        except :
            if not saveIfMurdered :
                raise
            exType, ex, tb = sys.exc_info()
            _dieGracefully(exType, tb)
            raise

    def run(self, **kwargs) :
        """Abtract function must be implemented in child. This function should implement the whole training process"""
        raise NotImplemented("Must be implemented in child")

class DefaultTrainer(Trainer_ABC) :
    """The default trainer should serve for most purposes"""

    def __init__(self,
        trainMaps,
        testMaps,
        validationMaps,
        model,
        ouputSchedule=Synchronous(),
        stopCriteria=[],
        onEpochStart=[],
        onEpochEnd=[],
        onStart=[],
        onEnd=[],
        ) :
        """
            :param DatasetMaps trainMaps: Layer mappings for the training set
            :param DatasetMaps testtrainMaps: Layer mappings for the testing set
            :param DatasetMaps validationMaps: Layer mappings for the validation set, if you do not wich to set one, pass None as argument
            :param int trainMiniBatchSize: The size of a training minibatch, use DefaultTrainer.ALL_SET for the whole set
            :param list stopCriteria: List of StopCriterion objects 
            :param int testMiniBatchSize: The size of a testing minibatch, use DefaultTrainer.ALL_SET for the whole set
            :param int validationMiniBatchSize: The size of a validationMiniBatchSize minibatch
            :param bool saveIfMurdered: Die gracefully in case of Exception or SIGTERM and save the current state of the model and logs
            :param string trainFunctionName: The name of the function to use for training
            :param string testFunctionName: The name of the function to use for testing
            :param string validationFunctionName: The name of the function to use for testing in validation
        """
        
        Trainer_ABC.__init__(self)

        self.maps = {
            "train": trainMaps
        }

        if testMaps is not None :
            self.maps["test"] = testMaps

        if validationMaps is not None :
            self.maps["validation"] = validationMaps

        self.stopCriteria = stopCriteria        
        self.onEpochStart = onEpochStart        
        self.onEpochEnd = onEpochEnd        
        self.onStart = onStart        
        self.onEnd = onEnd        
        self.model = model
        self.ouputSchedule = ouputSchedule
        self.startTime = None

    def start(self, runName) :
        """starts the training, cf. run() for the a description of the arguments"""
        self.startTime = time.time()
        for hook in self.onStart :
            hook.commit(self)

        Trainer_ABC.start( self, runName)
        
        for hook in self.onEnd :
            hook.commit(self)

    def run(self, name) :
        """
            :param str runName: The name of this run
            :param Recorder recorder: A recorder object
            :param int trainingOrder:
                * DefaultTrainer.SEQUENTIAL_TRAINING: Each output will be trained indipendetly on it's own epoch
                * DefaultTrainer.SIMULTANEOUS_TRAINING: All outputs are trained within the same epoch with the same inputs
                * Both are in O(m*n), where m is the number of mini batches and n the number of outputs
                * DefaultTrainer.RANDOM_PICK_TRAINING: Will pick one of the outputs at random for each example

            :param bool reset: Should the trainer be reset before starting the run
            :param dict moreHyperParameters: If provided, the fields in this dictionary will be included into the log .csv file
        """
        while True :
            for hook in self.onEpochStart :
                hook.commit(self)

            for mapName, aMap in self.maps.iteritems() :
                self.store["scores"][mapName] = self.ouputSchedule.run(aMap)
                
            for hook in self.onEpochEnd :
                hook.commit(self)

            self.store["runInfos"]["epoch"] += 1
            self.store["runInfos"]["runtime"] = (time.time() - self.startTime)

            for crit in self.stopCriteria :
                if crit.stop(self) :
                    raise MSTOP.EndOfTraining(crit)

