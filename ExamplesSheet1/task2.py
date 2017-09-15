import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math

class AsyncHopfield:

	def init(self,n, p, beta):
		self.m = 0
		self.n = n
		self.p = p
		self.time = 0
		self.beta = beta
		choices = [1,-1]
		patterns=[]
		for x in range (0,p):
			pattern = np.random.choice(choices,(n,1))
			patterns.append(pattern)
		states = np.random.choice(choices,(n,1))
		self.states = states
		self.patterns = patterns

	def calWeights(self):
		weights = np.zeros((self.n, self.n))
		for x in range(0,self.p):
			w = (self.patterns[x] * np.transpose(self.patterns[x]))
			weights = np.add(weights, w)
		weights = weights/self.n
		np.fill_diagonal(weights,0)
		self.weights = weights

	def feedInitPattern(self, index):
		self.initPattern = self.patterns[index]
		self.states = self.patterns[index]
		print(np.dot(self.states, np.transpose(self.initPattern)))

	def gFunc(self, b):
		return 1/(1 + math.exp(-2*self.beta* b))

	def updateRandNeuron(self):
		i = np.random.randint(0,self.n)
		bi = np.sum(np.dot(self.weights[i], self.states))
		
		prob = self.gFunc(bi)
		rand = np.random.random_sample()
		newValue = 1 if rand < prob else -1

		self.states[i] = newValue
		self.addMValue()

	def addMValue(self):
		self.m += (1/self.n)*(np.sum(np.dot(self.states, np.transpose(self.initPattern))))
		self.time += 1

	def TravellingM(self):
		avg = self.m/self.time
		return avg


N= 200
P = 5

network = AsyncHopfield()
network.init(N, P, 2)
network.calWeights()
network.feedInitPattern(0)
m = []
for x in range(0,15000):
	network.updateRandNeuron()
	if (x % 100 == 0) and x != 0:
		mean = network.TravellingM()
		m.append(mean)
		print(mean)

plt.plot(m)
plt.show()