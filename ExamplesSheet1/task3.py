import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randint


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
						temp = np.asarray(temp).T
						self.inputData.append(temp)
						self.targetOutput.append(float(word))
					counter += 1
		np.asarray(self.inputData)
		np.asarray(self.targetOutput)


	def initValues(self, learnR, beta):
		self.learnR = learnR
		self.beta = beta
		self.weights = (0.2 - (-0.2)) * np.random.random_sample((2,1)) + (-0.2)
		self.threshold = (1 - (-1)) * np.random.random_sample() + (-1)
		print(self.weights)


	def actFunc(self, b, deriv=False):
		if (deriv == True):
			return self.beta*(1-np.tanh(self.beta*b)**2)
		return np.tanh(self.beta*b)

	def trainNetwork(self, test=False):
		if test == True:
			rand = 1
		else:
			rand = randint(0,len(self.inputData)-1)
		# Forward propagation
		l0 = np.zeros((2,1))
		y = 0
		for x in self.inputData[rand]:
			l0.itemset((y,0),x)
			y+=1

		b = np.dot(np.transpose(self.weights), l0) - self.threshold
		l1 = self.actFunc(b)

		if test == True:
			print(l0)
			print(l1)

		#Backwards propagation
		# Error
		l1Error = self.targetOutput[rand] - l1

		# Delta error
		l1Delta = l1Error * self.actFunc(l1, True)

		self.weights = np.add(self.weights, self.learnR * np.dot(l0, l1Delta))
		self.threshold = np.add(self.threshold, -(self.learnR * l1Delta))

	def energyShit(self):
		sum = 0
		for x in range(0, len(self.inputData)):
			# Forward propagation
			l0 = np.zeros((2,1))
			y = 0
			for z in self.inputData[x]:
				l0.itemset((y,0),z)
				y+=1

			b = np.dot(np.transpose(self.weights), l0) - self.threshold
			output = self.actFunc(b)
			sum += (self.targetOutput[x] - output)**2
		return sum/2

learnR = 0.02
beta = 0.5

shitPlot = []
back = Backpropagation()
back.readData()
back.initValues(learnR, beta)
for x in range(1,100000):
	back.trainNetwork()
	shitPlot.append(back.energyShit())
	if x % 10000 == 0:
		print(x)
back.trainNetwork(True)
plt.plot(shitPlot)
plt.show()
