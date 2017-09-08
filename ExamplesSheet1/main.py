import matplotlib.pyplot as plt
import numpy as np


class Hopfield:

	def init(self,n, p):
		self.n = n
		self.p = p
		choices = [1,-1]
		patterns=[]
		for x in range (0,p):
			pattern = np.random.choice(choices,(n,1))
			patterns.append(pattern)
		states = np.random.choice(choices,(n,1))
		self.states = states
		self.patterns = patterns

	def calWeight(self):
		weights = np.zeros((self.n, self.n))
		for x in range(0,self.p):
			w = (self.patterns[x] * np.transpose(self.patterns[x]))/self.n
			weights = np.add(weights, w)
		self.weights = weights

	def feedInitPattern(self, index):
		self.initPattern = self.patterns[index]
		self.states = self.patterns[index]

	def stepNetwork(self):
		sign = lambda x: -1 if x < 0 else 1
		res = np.dot(self.weights, self.states)
		shp = res.shape
		res = np.fromiter((sign(xi) for xi in res), res.dtype)
		res = np.reshape( res, shp)
		self.states = res

	def getPError(self):
		result = np.add(np.divide(self.states, self.initPattern), 1)
		result = np.divide(result, 2)
		return np.sum(result)/self.n

hop = Hopfield();
hop.init(1000,137)
hop.calWeight()
hop.feedInitPattern(0)
hop.stepNetwork()
print(hop.getPError())
