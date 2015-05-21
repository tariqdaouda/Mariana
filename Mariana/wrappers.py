from collections import OrderedDict
import theano
import sys

import Mariana.candies as MCAN

class TheanoFunction(object) :
	"This class encapsulates a Theano function"

	def __init__(self, name, outputLayer, output_expressions, additional_input_expressions = {}, updates = [], **kwargs) :
		self.name = name
		self.outputLayer = outputLayer
		
		self.inputs = OrderedDict()
		self.tmpInputs = OrderedDict()
		for inp in self.outputLayer.network.inputs.itervalues() :
			self.inputs[inp.name] = inp.outputs
		self.inputs.update(additional_input_expressions)
		
		for i in self.inputs :
			self.tmpInputs[i] = None
		
		self.additional_input_expressions = additional_input_expressions
		self.outputs = output_expressions
		self.updates = updates
		self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.outputs, updates = self.updates, **kwargs)

		if any([x.__class__.__name__.lower().find("gpu") for x in self.theano_fct.maker.fgraph.toposort()]):
			device = "GPU"
		else:
			device = "CPU"
		MCAN.friendly("Run device", "I will use the [-%s-] to run function '%s' of layer '%s'!" % (device, name, outputLayer.name))

	def printGraph(self) :
		"""Print the theano graph of the function"""
		theano.printing.debugprint(self.theano_fct)

	def run(self, **kwargs) :
		for k in kwargs :
			self.tmpInputs[k] = kwargs[k]

		try :
			return self.theano_fct(*self.tmpInputs.values())
		except Exception as e :
			sys.stderr.write("!!=> Error in function '%s' for layer '%s':\n" % (self.name, self.outputLayer.name))
			sys.stderr.write("\t!!=> the arguments were:\n %s\n" % (kwargs))
			raise e

	def __call__(self, **kwargs) :
		return self.run(**kwargs)

	def __repr__(self) :
		return "<Mariana Theano Fct '%s'>" % self.name

	def __str__(self) :
		return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)