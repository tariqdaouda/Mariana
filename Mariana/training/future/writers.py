import sys, os, types
from pyGeno.tools.parsers.CSVTools import CSVFile

__all__ = ["Writer_ABC", "GGPlot"]

class Writer_ABC(object) :
    """A recorder is meant to be plugged into a trainer to record the
    advancement of the training. This is the interface a Recorder must expose."""
    
    def commit(self, trainer) :
        """Does something with the currenty state of the trainer's store and the model"""
        raise NotImplemented("Should be implemented in child")

class PrettyPrinter(Writer_ABC) :
    """"
    :param int printRate: The rate at which the status is printed on the console. If set to <= to 0, will never print.
    """

    def __init__(self, loggers, printRate=1, treeView=False):
        self.loggers = loggers
        self.printRate = printRate
        self.nbCommits = 0
        self.treeView = treeView
        self.tree = None

    def commit(self, trainer) :
        # def recTree(vals, res, value, i=0) :
        #     if i == len(vals)-1 :
        #         res.append("%s|->%s: %s" %(" "*i*2, vals[i], value) ) 
        #         return res

        #     s = "%s|-%s" %(" "*i*2, vals[i])
        #     res.append( s ) 
        #     return recTree(vals, res, value, i+1)

        def setTree(vals, res) :
            tres = res
            for v in vals :
                if v not in tres :
                    tres[v] = {}
                tres = tres[v]
            return res
       
        def compileTree(tree, res=[], currKey="", i=0) :
            for k, v in tree.iteritems() :
                key = "%s.%s" % (currKey,k)
                if v == {} :
                    res.append("%s|->%s: {%s}" %(" "*i*2, k, key) )
                else :
                    res.append("%s|-%s" %(" "*i*2, k))
                    res.extend(compileTree(v, res, key, i+1))

            return res

        s = ["\n>Epoch: %s, commit: %s, runtime: %s" %(trainer.store["runInfos"]["epoch"], self.nbCommits, trainer.store["runInfos"]["runtime"])]

        if self.treeView and self.tree is None :
            self.tree = {}
            for logi, log in enumerate(self.loggers) :
                self.tree[logi] = {}
                for k, v in log.log(trainer) :
                    self.tree[logi] = setTree(k.split("."), self.tree[logi])
                    # print k
                    #print self.tree[logi]
        #print self.tree
        #comp = compileTree(self.tree)
        #print len(comp)
        # print '\n'.join()
        #stop
         # lineSep = "%s\n"
        for log in self.loggers :
            try :
                name = log.name
            except :
                name = log.__class__.__name__

            sep = "-"*len(name)
            s.append("%s\n%s\n%s" % (sep, name, sep))
            for k, v in log.log(trainer) :
                if not self.treeView :
                    s.append("|-%s: %s" % (k, v))
                # else :
                    # s = recTree(k.split("."), s, v)

        print '\n'.join(s)
        sys.stdout.flush()
        self.nbCommits += 1
        
class CSV(Writer_ABC):
    """This training recorder will create a nice CSV (or tab delimited) file fit for using with ggplot2 and will update
    it as the training goes. It will also save the best model for each set of the trainer, and print
    regular reports on the console.

    :param string filename: The filename of the tsv to be generated. the extension '.ggplot2.tsv' will be added automatically
    :param int write: The rate at which the status is written on disk
    """

    def __init__(self, filename, loggers, separator="\t", writeRate=1):
        self.filename = filename.replace(".tsv", "") + ".ggplot.tsv"
        self.writeRate = writeRate
        self.loggers = loggers

        self.csvFile = None
        self.nbCommits = 0
        self.separator = separator

    def commit(self, trainer) :
        """"""
        
        if self.csvFile is not None :
            line = self.csvFile.newLine()
        else :
            line = {}
        
        for log in self.loggers :
            for k, v in log.log(trainer) :
                line[k] = v

        if self.csvFile is None :
            self.csvFile = CSVFile(legend = line.keys(), separator = self.separator)
            self.csvFile.streamToFile( self.filename, writeRate = self.writeRate )
            newLine = self.csvFile.newLine()
            for k, v in line.iteritems() :
                newLine[k] = v
            line = newLine

        self.nbCommits += 1    
        line.commit()

    def __len__(self) :
        """returns the number of commits performed"""
        return self.nbCommits
