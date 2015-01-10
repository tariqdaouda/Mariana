import json
import layers as L

class MarianaParser(object) :
	"parse a json entry to create network"
	def __init__(self) :
		pass

	def parseFile(self, jsonFile) :
		f = open(jsonFile)
		self.json = json.load(f)
		f.close()
		model = M.model(self.json['name'])

		for jl in jsonLayers :
			if jl['function'] == 'input' :
				layers.append(L.InputLayer(model = model, **jl))
			else :
				if jl['function'] == 'hidden' :
					layers.append(L.HiddenLayer(model = model, **jl))
				elif jl['function'] == 'output' :
					layers.append(L.OutputLayer(model = model, **jl))
				else :
					raise ValueError("Unknown layer function '%s'" % jl['function'])
		return model