import inspect, sys, os, shutil
import Mariana.HTML_Templates.template as MTMP


class Vulcan(MTMP.HTMLTemplate_ABC):
    """A theme"""
    TEMPLATE="""
        <!doctype html>
        <html>
        <head>
            <title>{TITLE}</title>
            {LIBS}
        </head>

        <body>
            <div id="all" class="uk-container uk-container-center">
                <div class="uk-grid uk-flex-center uk-text-center" uk-grid>
                    <div class="uk-card uk-card-default uk-width-expand">
                        <p>
                            <a target="_blank" href='{DOCUMENTATION_URL}'>
                                <img src="{LIBS_FOLDER}/mariana_logo.png" alt="Documentation" height="150" width="150" title="Documentation"/>
                            </a>
                        </p> 
                        <h3 class="uk-card-title uk-margin-remove-bottom">{TITLE}</h3>
                        <p class="uk-text-meta uk-margin-remove-top"><time datetime="{MACHINE_TIME}">{USER_TIME}</time></p>
                        <div class="uk-card-body">
                            {MODEL_NOTES}
                        </div>
                        <a target="_blank" href="{GITHUB_URL}" class="uk-icon-button">
                            <i uk-icon="icon: github"></i>
                        </a>
                    </div>
                    <div class="uk-width-1-1"></div>
                    <div class="uk-width-1-1">
                        <graph-view nodesp='{LAYERS_JSON}' edgesp='{EDGES_JSON}' physicsp="false" ></graph-view>
                    </div>
                </div>
            </div>
        </body>

        <script src="{LIBS_FOLDER}/uikit/3.0.0-beta.21/js/uikit-icons.min.js"></script>
        <script type="text/javascript" src="{LIBS_FOLDER}/vulcan.js"></script>

        </html>
    """

    def __init__(self):
        super(Vulcan, self).__init__()
        self.dirname = os.path.dirname(inspect.getfile(sys.modules[__name__]))
        
        # f = open(os.path.join(self.dirname, "vulcan.html"))
        # self.html = f.read()
        # f.close()
        
        self.weblibsDir = os.path.join(self.dirname, "weblibs")
        self.libs_local = """
            <script type="text/javascript" src="{LIBS_FOLDER}/jquery-3.2.1.min.js"></script>
            <link rel="stylesheet" href="{LIBS_FOLDER}/uikit/3.0.0-beta.21/css/uikit.min.css" />
            <script src="{LIBS_FOLDER}/uikit/3.0.0-beta.21/js/uikit.min.js"></script>
            <script type="text/javascript" src="{LIBS_FOLDER}/vue.min.js"></script>    
            <script type="text/javascript" src="{LIBS_FOLDER}/vis/4.19.1/vis.min.js"></script>
            <link href="{LIBS_FOLDER}/vis/4.19.1/vis.min.css" rel="stylesheet" type="text/css"/>
            <link href="{LIBS_FOLDER}/vulcan.css" rel="stylesheet" type="text/css"/>
        """

        self.libs_remote = """
            <script type="text/javascript" src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.0-beta.21/css/uikit.min.css" />
            <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.0-beta.21/js/uikit.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/uikit/3.0.0-beta.21/js/uikit-icons.min.js"></script>
            <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.19.1/vis.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.19.1/vis.min.css" rel="stylesheet" type="text/css"/>
            <script type="text/javascript" src="https://unpkg.com/vue"></script>
        """

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

    def render(self, filename, networkJson, save) :
        import time
        import json

        title = os.path.basename(filename)
        noSpaceTitle = title.replace(" ", "-")
        libsFolder = "%s_weblibs" % noSpaceTitle
    
        if save :
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

        if save :
            libs = self.libs_local
        else :
            libs = self.libs_remote

        html = self.TEMPLATE.format(
            TITLE=title,
            LIBS=libs,
            LIBS_FOLDER=libsFolder,
            DOCUMENTATION_URL="http://bioinfo.iric.ca/~daoudat/Mariana/",
            GITHUB_URL="https://github.com/tariqdaouda/Mariana",
            MODEL_NOTES=self.formatNotes(networkJson["notes"]),
            MACHINE_TIME=time.time(),
            USER_TIME=time.ctime().replace("_", " "),
            LAYERS_JSON=json.dumps(layers),
            EDGES_JSON=json.dumps(networkJson["edges"])
        )
        
        if save :
            if os.path.isdir(self.weblibsDir) :
                dstFolder = os.path.join(currFolder, libsFolder)
                if os.path.isdir(dstFolder) :
                    shutil.rmtree(dstFolder)    
                shutil.copytree(self.weblibsDir, dstFolder)
            
            f = open(os.path.join(currFolder, "%s.html" % noSpaceTitle), "w") 
            f.write(html)
            f.close()

        return html