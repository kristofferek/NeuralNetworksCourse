import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math

class Backpropagation:
	def readData(self):
		self.inputData = []
		self.targetOutput = []
		with open('training_data.txt','r') as f:
			for line in f:
				counter = 0
				temp = []
				for word in line.split():
					if counter < 2:
						temp.append(word)
					else:
						self.inputData.append(temp)
						self.targetOutput.append(word)
					counter += 1

	def initValues(self, learnR, beta):
		self.learnR = learnR
		self.beta = beta 
		np.random.seed(1)
		self.weights = (0.2 - (-0.2)) * np.random.random_sample((2,1)) + (-0.2)
		self.weights = map(float, self.weights)
		self.thresholds = (1 - (-1)) * np.random.random_sample((2,1)) + (-1)

	def actFunc(self, b, deriv=False):
		if (deriv == True):
			return self.beta*(1-np.tanh(self.beta*b)**2)
		return np.tanh(self.beta*b)

	def trainNetwork(self):
		rand = np.random.randint(0,len(self.inputData))
		# Forward propagation
		l0 = self.inputData[rand]
		l0 = map(float, l0)
		l1 = self.actFunc(np.dot(self.weights, l0))

		#Backwards propagation
		# Error
		l1Error = self.targetOutput[rand] - l1

		# Delta error
		l1Delta = l1Error * self.actFunc(l1, True)

		self.weights += self.learnR * np.dot(l0.T, l1Delta)

learnR = 0.02
beta = 0.5


back = Backpropagation()
back.readData()
back.initValues(learnR, beta)
back.trainNetwork()
