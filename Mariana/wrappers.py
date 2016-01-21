from collections import OrderedDict
import theano, sys, numpy
import sys

import Mariana.candies as MCAN
import Mariana.settings as MSET

DEVICE_IS_GPU = (theano.config.device.find("gpu") > -1)

class TheanoFunction(object) :
	"""
	This class encapsulates a Theano function.
	TheanoFunction objects should be defined as self attributes in the setCustomTheanoFunctions() function of output layers.
	It will also generate custom error messages whose verbosity depends on Mariana.settings.VERBOSE. Set it to False to get quieter
	error messages.
	"""

	def __init__(self, name, outputLayer, output_expressions, additional_input_expressions = {}, updates = [], **kwargs) :
		"""
		:param str name: name of the function
		:param Output outputLayer: the output layer the function should be applied to
		:param list output_expressions: list of the symbolic expressions you want as output
		:param dict additional_input_expressions: additional inputs needed to compute the expressions
		:param list updates: list of tuples (shared variable, symbolic expression of the update to be applied to it)
		:param dict \*\*kwargs: additional arguments to passed to the real theano function underneath
		"""
		self.cast_warning_told = False

		self.name = name
		self.outputLayer = outputLayer

		self.inputs = OrderedDict()
		inpSet = set()
		for inp in self.outputLayer.network.inputs.itervalues() :
			inpOut = inp.inputs

			if inpOut not in inpSet :
				self.inputs[inp.name] = inpOut
				inpSet.add(inpOut)

		for k, v in additional_input_expressions.iteritems() :
			if v not in inpSet :
				self.inputs[k] = v
				inpSet.add(v)

		self.fctInputs = OrderedDict()
		for i in self.inputs :
			self.fctInputs[i] = None

		self.additional_input_expressions = additional_input_expressions
		self.outputs = output_expressions
		self.updates = updates

		self.theano_fct = theano.function(inputs = self.inputs.values(), outputs = self.outputs, updates = self.updates, **kwargs)

		warningMsg = False
		if DEVICE_IS_GPU :
			device = "GPU"
			msg = "I will use the [-%s-] to run function '%s' of layer '%s'!" % (device, name, outputLayer.name)
			if str(self.getToposort()).find("float64") > -1:
				warningMsg = True
				msg += "\n\nBut there are some float64s that do not fit on the GPU and will slow down the computations.\nPlease consider:"
				msg += "\n\t* Launching with THEANO_FLAGS=device=gpu,floatX=float32 python <your script>.py."
				msg += "\n\t* If you have any dmatrix, dvector or dscalar in your code replace them with matrix, vector, scalar."
		else:
			device = "CPU"
			msg = "I will use the [-%s-] to run function '%s' of layer '%s'!" % (device, name, outputLayer.name)


		MCAN.friendly("Run device", msg, warning = warningMsg)
		

	def getToposort(self) :
		"""returns the toposort ( name of all ops  of the function in order of application ) of the function"""
		return self.theano_fct.maker.fgraph.toposort()

	def printGraph(self) :
		"""Print the theano graph of the function"""
		theano.printing.debugprint(self.theano_fct)

	def dumpToImage(self, filename) :
		"""saves the function graph into an image file, requires pydot"""
		theano.printing.pydotprint(self.theano_fct, filename)

	def dumpToHTML(self, filename) :
		"""saves the function graph into an interactive html file, requires pydot and potentially an internet connection"""
		import theano.d3viz as d3v
		d3v.d3viz(self.theano_fct, filename)

	def run(self, **kwargs) :

		def _die(fctName, outputLayer, kwargs, exc) :
			sys.stderr.write("!!=> Error in function '%s' for layer '%s':\n" % (fctName, outputLayer.name))
			sys.stderr.write("\t!!=> the arguments were:\n %s\n" % (kwargs))
			raise exc

		self.fctInputs.update(kwargs)
		return self.theano_fct(*self.fctInputs.values())

	def __call__(self, **kwargs) :
		return self.run(**kwargs)

	def __repr__(self) :
		return "<Mariana Theano Fct '%s'>" % self.name

	def __str__(self) :
		return "<Mariana Theano Fct '%s': %s>" % (self.name, self.theano_fct)
