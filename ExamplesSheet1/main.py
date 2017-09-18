import matplotlib.patches as mpatches
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
			w = (self.patterns[x] * np.transpose(self.patterns[x]))
			weights = np.add(weights, w)
		self.weights = weights/self.n

	def feedInitPattern(self, index):
		self.initPattern = np.copy(self.patterns[index])
		self.states = np.copy(self.patterns[index])

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

def getAvgPError(n, p, iterations):
	hop = Hopfield()
	nCorrect = 0
	nTotal = 0
	for x in range(0,iterations):
		hop.init(n,p)
		hop.calWeight()
		for _p in range(p):
			hop.feedInitPattern(_p)
			hop.stepNetwork()
			nCorrect += hop.getPError()
			nTotal += n
	print(hop.weights)
	return 1- (nCorrect/nTotal)

def getTheoreticalPError(n, p):
	return (1 - math.erf(np.sqrt(n/(2*p)) + np.sqrt(p/(2*n))))/2

bitsToSee = 100000
N = 200
P = [1]

x = np.divide(P, N)
yReal = []
yTheory = []
for p in P:
	iterations = math.ceil(bitsToSee / (p*N))
	realPerror = getAvgPError(N, p, iterations)
	theoryPerror = getTheoreticalPError(N,p)
	print(p)
	print('Real: ', realPerror)
	print('Theory: ',  theoryPerror)
	print('\n')
	yReal.append(realPerror)
	yTheory.append(theoryPerror)

print('Calculations done')
plt.plot(x, yReal, 'r-', x, yTheory, 'g-')
red_patch = mpatches.Patch(color='red', label='Real data')
legend1 = plt.legend(handles=[red_patch], loc=1)
plt.gca().add_artist(legend1)
green = mpatches.Patch(color='green', label='Theoretical data')
plt.legend(handles=[green], loc=2)
plt.xlabel('p/N')
plt.ylabel('Perror')

plt.show()