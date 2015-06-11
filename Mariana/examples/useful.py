def load_mnist() :
	"""If i can't find it i will attempt to download it from LISA's place"""
	import urllib, os, gzip, cPickle
	dataset = 'mnist.pkl.gz'
	if (not os.path.isfile(dataset)):
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print '==> Downloading data from %s' % origin
		urllib.urlretrieve(origin, dataset)

	f = gzip.open(dataset, 'rb')
	return cPickle.load(f)