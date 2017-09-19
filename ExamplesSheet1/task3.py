import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randint
import time

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
						temp = np.asarray(temp)
						self.inputData.append(temp)
						self.targetOutput.append(float(word))
					counter += 1
		self.inputData = np.asarray(self.inputData)
		self.targetOutput = np.asarray(self.targetOutput)


	def initValues(self, learnR, beta):
		self.learnR = learnR
		self.beta = beta

		# 2x1 matrix, random values [-0.2, 0.2]
		self.weights = (0.2 - (-0.2)) * np.random.random_sample((2,1)) + (-0.2)

		# single random value [-1, 1]
		self.threshold = (1 - (-1)) * np.random.random_sample() + (-1)


	def actFunc(self, b, deriv=False):
		if (deriv == True):
			return self.beta*(1-np.square(np.tanh(self.beta*b)))
		return np.tanh(self.beta*b)

	def trainNetwork(self, test=False):
		if test == True:
			rand = 1
		else:
			rand = randint(0,len(self.inputData)-1)
		# Forward propagation
		l0 = (self.inputData[rand])[np.newaxis].T

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
		l1Delta = self.learnR * l1Delta
		#Updates
		self.weights = np.add(self.weights, np.dot(l0, l1Delta))
		self.threshold = np.add(self.threshold, -l1Delta)

	def energyFunc(self):
		# Forward propagation
		b = np.dot(self.inputData, self.weights) - self.threshold
		output = self.actFunc(b)
		sum = np.sum(np.square(self.targetOutput - output))
		return sum/2

learnR = 0.02
beta = 0.5

shitPlot = []
back = Backpropagation()
back.readData()
back.initValues(learnR, beta)
start = time.time()
for x in range(1,100000):
	back.trainNetwork()
	shitPlot.append(back.energyFunc())

	if x % 10000 == 0:
		print(x)
		end = time.time()
		print(end - start, ' seconds since begining')

back.trainNetwork(True)
plt.plot(shitPlot)
plt.show()
