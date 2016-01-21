Running on GPU
================

Mariana is fully capable of running on both GPU and CPU and running Mariana on GPU is the same as running Theano on GPU.
This is a quick tutorial on how to get you started, if you need more information you can have a look at the documention of Theano_ .

.. _Theano: http://deeplearning.net/software/theano/tutorial/using_gpu.html

To make sure your machine is GPU enabled, go to the examples folder and run::

	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python theano_device_check1.py

You can also run it on CPU mode to compare the performances::

	THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python theano_device_check1.py

To run Mariana on GPU mode you can use::

	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python <my script>.py

It is also a good practice to make sure your data is in float32 (as GPUs cannot handle more). Because Mariana has your best interest at heart,
she will tell you on wich device she is running the functions (unless you set *settings.VERBOSE* to *False*). She also will warn you if you specify that
you want to use the GPU but have some float64s that slow down the performances.