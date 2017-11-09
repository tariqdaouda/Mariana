import inspect, sys, os, shutil
import Mariana.HTML_Templates.template as MTMP

class Vulcan(MTMP.HTMLTemplate_ABC):
    """A theme"""
    def __init__(self):
        super(Vulcan, self).__init__()
        self.dirname = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        
        f = open(os.path.join(self.dirname, "vulcan.html"))
        self.html = f.read()
        f.close()
        
        self.weblibsDir = os.path.join(self.dirname, "weblibs")
    
    def formatNotes(self, notes) :
        tmp = """
            <div class="uk-flex-center uk-child-width-1-2@s uk-child-width-1-3@m" uk-grid>
               {trs} 
            </div>
        """

        tmpTrs ="""
            <div class="uk-card uk-card-default uk-card-small uk-card-body">
                <h3 class="uk-card-title">{title}</h3>
                <p>{text}</p>
            </div>
        """
        
        trs = []
        for k, v in notes.iteritems() :
            trs.append(tmpTrs.format(title = k, text = v))
       
        return tmp.format(trs = "\n".join(trs))

    def render(self, filename, networkJson) :
        import time
        import json

        title = os.path.basename(filename)
        noSpaceTitle = title.replace(" ", "-")
        libsFolder = "%s_weblibs" % noSpaceTitle
        currFolder = os.path.dirname(filename)

        layers = []
        for l in networkJson["layers"] :
            absCheck = set()
            dct = {"name": l, "shape": networkJson["layers"][l]['shape'], "level": networkJson["layers"][l]['level'], "abstractions": {"size": 1, "layer":[{"name": "Class", "value": networkJson["layers"][l]['class']}]} }
            for cat in ["parameters", "hyperParameters", "notes"] :
                dct[cat] = {"size": 0}
                dct[cat]["layer"] = []
                for pName, pVal in networkJson["layers"][l][cat].iteritems() :
                    if cat == "notes" :
                        pKey = pName
                    else :
                        pKey = "%s.%s" % (l, pName)

                    dct[cat]["layer"].append({"name": pKey, "value": pVal})
                    dct[cat]["size"] += 1
                
                for absCat, abstractions in networkJson["layers"][l]["abstractions"].iteritems() :
                    dct[cat][absCat] = []
                    if absCat not in dct["abstractions"] :
                        dct["abstractions"][absCat] = []
    
                    for absName, absVal in abstractions.iteritems() :
                        if absName not in absCheck :
                            absCheck.add(absName)
                            dct["abstractions"]["size"] += 1
                            dct["abstractions"][absCat].append({"name": absName, "value": ""})
                        
                        try :
                            for pName, pVal in absVal[cat].iteritems() :
                                if cat == "notes" :
                                    pKey = pName
                                else :
                                    pKey = "{absName}.{pName}".format(absName = absName, pName = pName)
                                
                                dct[cat][absCat].append({"name": pKey, "value": pVal})
                                dct[cat]["size"] += 1
                        except KeyError :
                            pass    
            layers.append([l, dct])

        html = self.html.format(
            TITLE=title,
            LIBS_FOLDER=libsFolder,
            DOCUMENTATION_URL="http://bioinfo.iric.ca/~daoudat/Mariana/",
            GITHUB_URL="https://github.com/tariqdaouda/Mariana",
            MODEL_NOTES=self.formatNotes(networkJson["notes"]),
            MACHINE_TIME=time.time(),
            USER_TIME=time.ctime().replace("_", " "),
            LAYERS_JSON=json.dumps(layers),
            EDGES_JSON=json.dumps(networkJson["edges"])
        )
        
        
        if os.path.isdir(self.weblibsDir) :
            dstFolder = os.path.join(currFolder, libsFolder)
            if os.path.isdir(dstFolder) :
                shutil.rmtree(dstFolder)    
            shutil.copytree(self.weblibsDir, dstFolder)
        
        f = open(os.path.join(currFolder, "%s.html" % noSpaceTitle), "w") 
        f.write(html)
        f.close()