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
		self.initPattern = np.copy(self.patterns[index])
		self.states = np.copy(self.patterns[index])

	def gFunc(self, b):
		return 1/(1 + math.exp(-2*self.beta* b))

	def updateRandNeuron(self):
		i = np.random.randint(0,self.n)
		bi = np.dot(self.weights[i], self.states)[0]

		prob = self.gFunc(bi)
		rand = np.random.random_sample()
		newValue = 1 if rand < prob else -1

		self.states.itemset((i, 0), newValue)
		self.addMValue()

	def addMValue(self):
		nCorrect = np.dot(np.transpose(self.states), self.initPattern)[0][0]
		self.m += (1/self.n)*(nCorrect)
		self.time += 1

	def TravellingM(self):
		avg = self.m/self.time
		return avg


N= 200
P = 5

mCollection = [] 

for i in range(0,20):
	print(i)
	network = AsyncHopfield()
	network.init(N, P, 2)
	network.calWeights()
	network.feedInitPattern(0)
	m = [1]
	for x in range(500000):
		network.updateRandNeuron()
		if (x % 100 == 0) and x != 0:
			mean = network.TravellingM()
			m.append(mean)
	mCollection.append(m)

for m in mCollection:
	plt.plot(m)

plt.ylim(0,1)
plt.show()
