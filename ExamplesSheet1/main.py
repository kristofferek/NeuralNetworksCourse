import matplotlib.pyplot as plt
import numpy as np
import math

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
		return np.sum(result)

def getAvgPError(n, p, avgSize):
	hop = Hopfield()
	nCorrect = 0
	nTotal = 0
	for x in range(0,avgSize):
		hop.init(n,p)
		hop.calWeight()
		for _p in range(p):
			hop.feedInitPattern(_p)
			hop.stepNetwork()
			nCorrect += hop.getPError()
			nTotal += n
			
	return 1- (nCorrect/nTotal)

def getTheoreticalPError(n, p):
	return (1 - math.erf(np.sqrt(n/(2*p))))/2

bitsToSee = 100000
N = 200
P = [1]
for i in range(1, 21):
	P.append(i*20)

x = np.divide(P, N)
yReal = []
yTheory = []
for p in P:
	iterations = math.ceil(bitsToSee / (p*N))
	yReal.append(getAvgPError(N, p, iterations))
	yTheory.append(getTheoreticalPError(N,p))

plt.plot(x, yReal, 'r-', x, yTheory, 'g-')
plt.show()