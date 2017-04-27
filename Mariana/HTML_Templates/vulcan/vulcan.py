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
        
        self.jsFP = os.path.join(self.dirname, "vulcan.js")
        f = open(self.jsFP)
        self.js = f.read()
        f.close()
        
        self.cssFP = os.path.join(self.dirname, "vulcan.css")
        f = open(self.cssFP)
        self.css = f.read()
        f.close()

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
        currFolder = os.path.dirname(filename)

        html = self.html.format(
            TITLE=title,
            MODEL_NOTES=self.formatNotes(networkJson["notes"]),
            MACHINE_TIME=time.time(),
            USER_TIME=time.ctime().replace("_", " "),
            LAYERS_JSON=json.dumps(networkJson["layers"]).replace('"', "'"),
            EDGES_JSON=json.dumps(networkJson["edges"]).replace('"', "'")
        )
        
        webFolder = "%s_web" % title
        if not os.path.exists(webFolder) :
            os.mkdir(webFolder)

        shutil.copy(self.jsFP, os.path.join(currFolder, webFolder, "vulcan.js"))
        shutil.copy(self.cssFP, os.path.join(currFolder, webFolder, "vulcan.css"))
        
        f = open(os.path.join(currFolder, "%s.html" % title), "w") 
        f.write(html)
        f.close()