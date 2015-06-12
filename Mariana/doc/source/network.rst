Networks
===============

Mariana Networks are graphs of connected layers and are in charge of calling the behind the scenes Theano functions.
Networks can also be saved into files, reloaded, and exported to DOT format so they can be visualized in a DOT visualizer such as graphviz.
Networks are not supposed to be manually instanciated, but are automatically created when layers are connected. 
You can forget about all the graph manipulation functions and simply use the shorthand **">"** to connect layer together.

.. automodule:: Mariana.network
   :members:

