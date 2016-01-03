import mnist_basics as MNIST      #reading in MNIST files
import theano                     #theano
import theano.tensor as T         #theano methods
import theano.tensor.nnet as Tann #neural net methods
import numpy as np                #numpy
import numpy.random as rng        #random number generator
#
def calc_avg_vect_dist(vectors):
	n = len(vectors); sum = 0
	for i in range(n):
		for j in range(i+1,n):
			sum += vector_distance(vectors[i],vectors[j])
	return 2*sum/(n*(n-1))
#
class ANN:
	def __init__(self, innput, hidden_layers, lr):
		self.innput = innput
		self.make_ann(hidden_layers, lr)
		#
	def make_ann(self, hidden_layers, lr):
		self.W = [theano.shared(
					rng.uniform(-0.1,0.1,size=(784,hidden_layers[0]) 
					)
				)]
		self.B = [theano.shared(
					rng.uniform(-0.1,0.1,size=(784) 
					)
				)]
		innput = T.vector('innput')
		self.X = [Tann.sigmoid(T.dot(innput, self.W[0])+self.B[0])]
		params = [self.W[0], self.B[0]]
		for n in range(1,len(hidden_layers)):
			#Finding number of inputs
			n_in = hidden_layers[n-1]
			n_out = hidden_layers[n]
			#making Bias and weights for a layer
			self.W.append(theano.shared(
					rng.uniform(-0.1,0.1,size=(n_in,n_out) 
					)
				)
			)
			#
			self.B.append(
				theano.shared(
					rng.uniform(-0.1,0.1,size=(n_in) 
					)
				)
			)
			#
			self.X.append(
				Tann.sigmoid(T.dot(self.W[n],self.W[n-1])+self.B[n])
			)
			params.append(self.W[n])
			params.append(self.B[n])
		#
		error = T.sum((innput-self.W[-1])**2)
		print (error)
		print (params)
		#
		
		gradients = T.grad(error,params)
	
		backprop_acts = [
			(p, p - self.lrate*g) for p,g in zip(params,gradients)]
		self.predictor = theano.function([innput],[self.X])
		self.trainer = theano.function(
				[innput],error,updates=backprop_acts
			)
	#
	def do_training(self,epochs=100,test_interval=None):
		errors = []
		if test_interval: self.avg_vector_distances = []
		for i in range(epochs):
			error = 0
			for c in self.innput:
				error += self.trainer(c)
			errors.append(error)
			if test_interval: self.consider_interim_test(i,test_interval)
	#
	def do_testing(self,scatter=True):
		hidden_activations = []
		for c in self.innput:
			_,hact = self.predictor(c)
			hidden_activations.append(hact)
		return hidden_activations
	#
	def consider_interim_test(self,epoch,test_interval):
		if epoch % test_interval == 0:
			self.avg_vector_distances.append(calc_avg_vect_dist(self.do_testing(scatter=False)))
#
def main():
	images, labels = MNIST.gen_flat_cases(digits=np.arange(10),type='training',cases=(MNIST.load_mnist(dataset="training", digits=np.arange(10), path="datasets/")))
	images = np.divide(images,255)

	ann = ANN(images, [10, 10], 0.1)
	ann.do_training(10, test_interval=10)
	return ann.do_testing()
#
if __name__ == '__main__':
	l = main()
	print(l)