import matplotlib.pyplot as plt
import cPickle, sys

c = cPickle.load( open(sys.argv[1]))

plt.plot(c['train'], label = "train")
plt.plot(c['validation'], label = "validation")
plt.plot(c['test'], label = "test")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)

In [7]: plt.show()
