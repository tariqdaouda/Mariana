__all__ = ["Abstraction_ABC"]

class Abstraction_ABC(object):
 	"""
 	This class represents a layer modifier. This class must includes a list attribute **self.hyperParameters** containing the names of all attributes that must be considered
 	as hyper-parameters.
 	"""
	def __init__(self, *args, **kwargs):
		self.name = self.__class__.__name__
 		self.hyperParameters = []
 		self.parameters = {}

	def apply(self, layer, cost) :
		"""Apply to a layer and update networks's log"""
		raise NotImplemented("Must be implemented in child")
		
	def toJson(self) :
		"""A json representation of the object"""

		res = {
			"class": self.name,
			"hyperParameters": {}
		}
		for h in self.hyperParameters :
			res["hyperParameters"][h] = getattr(self, h)
		
		return res

	def __repr__(self) :
		return "< %s >" % self.name