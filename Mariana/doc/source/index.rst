.. Mariana documentation master file, created by
   sphinx-quickstart on Thu Jun 11 13:02:52 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Mariana: Deep Neural Networks should be Easy to Write
=====================================================

Named after the deepest place on earth (Mariana trench), Mariana is a Machine Learning Framework built on top of Theano, that focuses on ease of use.  Mariana lives on github_.

.. _github: https://github.com/tariqdaouda/Mariana/

Why is it cool?
----------------

**If you can draw it, you can write it.**

Mariana provides an interface so simple and intuitive that writing models becomes a breeze.
Networks are graphs of connected layers and that allows for the craziest deepest architectures 
you can think of, as well as for a super light and clean interface. The paradigm is simple
create layers and connect them using **'>'**.
Plugging per layer regularizations, costs and mofications such as dropout and custom initilisations
is also super easy, and Mariana also supports trainers that encapsulate the whole training to make things
even easier.

So in short:
  
  * no YAML
  * completely modular and extendable
  * use the trainer to encapsulate your training in a safe environement
  * write your models super fast
  * save your models and resume training
  * export your models into DOT format to obtain clean and easy to communicate graphs
  * free your imagination and experiment
  * no requirements concerning the format of the datasets


What can you do with Mariana
----------------------------

Any type of deep, shallow, feed-forward, back-prop trained Neural Network. Convolutional Nets and Recurrent Nets are not supported yet but they will be.


A word about the **'>'**
-------------------------

When communicating about neural networks people often draw sets of connected layers. That's the idea behind Mariana: layers are first defined, then connected using the **'>'** operator. 

Can it run on GPU?
------------------

At the heart of Mariana are Theano functions, so the answer is yes. The guys behind Theano really did an awesome
job of optimization, so it should be pretty fast, wether you're running on CPU or GPU.


Extendable
----------

Mariana is an extendable framework that provides abstract classes allowing you to define new types of layers, learning scenarios, costs, stop criteria, recorders and trainers. Feel free to taylor it to your needs.

Installation
=============

Clone it from git!::

	git clone https://github.com/tariqdaouda/Mariana.git
	cd Mariana
	python setup.py develop

**Upgrade**::

	git pull

Contents:

.. toctree::
   :maxdepth: 2

   intro
   training
   layers
   network
   costs
   scenari
   activation
   decorators
   regularisation
   wrappers
   candies
   tips

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

