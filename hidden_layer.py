import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import theano.tensor.nnet as Tann
#
print (np.random.uniform(-0.1,0.1,size=(2,5)))
class hidden_layer:
	def __init__(n_weights, n_nodes, innput):
		self.n_weights = n_weights
		self.n_nodes = n_nodes
		self.innput = innput
		W_vals = numpy.asarray(rng.uniform(-0.1,0.1), size=(n_weights, n_nodes))
		W = theano.shared(value=W_vals, name='W')
		self.W=W
		#
		B_vals = numpy.asarray(rng.uniform(-0.1,0.1), size=n_nodes)
		B = theano.shared(value=B_vals, name='B')
		self.B=B
		#
	def Foo(self):
		self.x = Tann.sigmoid(T.dot(innput, self.W)+self.B)
		self.params = [self.W,self.B]