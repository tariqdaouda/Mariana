class HTMLTemplate_ABC(object):
    """The class that all templates must follow"""
    def __init__(self):
        super(HTMLTemplate, self).__init__()

    def render(self, filename) :
        """write the thing to disk"""
        raise NotImplementedError("Should be implemented in child: %s" % self.name)
        