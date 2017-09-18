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
						temp.append(float(word))
					else:
						self.inputData.append(temp)
						self.targetOutput.append(float(word))
					counter += 1
		np.asarray(self.inputData)
		np.asarray(self.targetOutput)


	def initValues(self, learnR, beta):
		self.learnR = learnR
		self.beta = beta
		np.random.seed(1)
		self.weights = (0.2 - (-0.2)) * np.random.random_sample((2,1)) + (-0.2)
		self.thresholds = (1 - (-1)) * np.random.random_sample((2,1)) + (-1)

	def actFunc(self, b, deriv=False):
		if (deriv == True):
			return self.beta*(1-np.tanh(self.beta*b)**2)
		return np.tanh(self.beta*b)

	def trainNetwork(self):
		rand = np.random.randint(0,len(self.inputData))
		# Forward propagation
		l0 = np.zeros((2,1))
		for x in self.inputData[rand]:
			y = 0
			l0.itemset((y,0),x)
			y+=1
		l1 = self.actFunc(np.dot(np.transpose(self.weights), l0))
		print(l1)

		#Backwards propagation
		# Error
		l1Error = self.targetOutput[rand] - l1

		# Delta error
		l1Delta = l1Error * self.actFunc(l1, True)

		self.weights += self.learnR * np.dot(l0, l1Delta)
		print(self.weights)
learnR = 0.02
beta = 0.5


back = Backpropagation()
back.readData()
back.initValues(learnR, beta)
back.trainNetwork()
