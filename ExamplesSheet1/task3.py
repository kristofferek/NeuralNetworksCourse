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
						temp = np.asarray(temp)
						self.inputData.append(temp)
						self.targetOutput.append(float(word))
					counter += 1
		self.inputData = np.asarray(self.inputData)
		self.targetOutput = np.asarray(self.targetOutput)

		# Validation data
		self.V_inputData = []
		self.V_targetOutput = []
		with open('valid_data.txt','r') as f:
			for line in f:
				counter = 0
				temp = []
				for word in line.split():
					if counter < 2:
						temp.append(float(word))
					else:
						temp = np.asarray(temp)
						self.V_inputData.append(temp)
						self.V_targetOutput.append(float(word))
					counter += 1
		self.V_inputData = np.asarray(self.V_inputData)
		self.V_targetOutput = np.asarray(self.V_targetOutput)


	def initValues(self, learnR, beta):
		self.learnR = learnR
		self.beta = beta

		# 2x1 matrix, random values [-0.2, 0.2]
		self.weights = (0.2 - (-0.2)) * np.random.random_sample((2,1)) + (-0.2)

		# single random value [-1, 1]
		self.threshold = (1 - (-1)) * np.random.random_sample() + (-1)


	def actFunc(self, b, deriv=False):
		if (deriv == True):
			return self.beta*(1-np.tanh(self.beta*b)**2)
		return np.tanh(self.beta*b)

	def trainNetwork(self):
		rand = randint(0,len(self.inputData)-1)
		# Forward propagation
		l0 = (self.inputData[rand])[np.newaxis].T

		b = np.dot(np.transpose(self.weights), l0) - self.threshold

		l1 = self.actFunc(b)

		#Backwards propagation
		# Error
		l1Error = self.targetOutput[rand] - l1

		# Delta error
		l1Delta = l1Error * self.actFunc(l1, True)
		l1Delta = self.learnR * l1Delta
		#Updates
		self.weights = np.add(self.weights, np.dot(l0, l1Delta))
		self.threshold = np.add(self.threshold, -l1Delta)

	def energyFunc(self, input, target):
		# Forward propagation
		b = np.dot(input, self.weights) - self.threshold
		output = self.actFunc(b)
		sum = np.sum(np.square(target[np.newaxis].T - output))
		return sum/2

	def classError(self, input, target):
		b = np.dot(input, self.weights) - self.threshold
		output = self.actFunc(b)
		sum = np.sum(np.absolute(target[np.newaxis].T - np.sign(output)))
		return sum * (1/(2*len(input)))


learnR = 0.02
beta = 0.5
C_T = []
C_V = []
trainingEnergy = []
validationEnergy = []

# Experiments
for experiment in range(0,10):
	t = []
	v = []
	back = Backpropagation()
	back.readData()
	back.initValues(learnR, beta)

	# Learning iterations
	for i in range(1,100000):
		back.trainNetwork()
		t.append(back.energyFunc(back.inputData, back.targetOutput))
		v.append(back.energyFunc(back.V_inputData, back.V_targetOutput))
	
	print('Experiment: ', experiment+1)
	trainingEnergy.append(t)
	validationEnergy.append(v)

	# Training classification error
	cErr = back.classError(back.inputData, back.targetOutput)
	C_T.append(cErr)

	# Validation classification error
	cErr = back.classError(back.V_inputData, back.V_targetOutput)
	C_V.append(cErr)

# Classification error prints
np.array(C_T)
np.array(C_V)
print('Training Avg: ', np.average(C_T))
print('Training Mini: ', np.amin(C_T))
print('Training Var: ', np.var(C_T))

print('Validation Avg: ', np.average(C_V))
print('Validation Mini: ', np.amin(C_V))
print('Validation Var: ', np.var(C_V))

# Plotting
f, axarr = plt.subplots(2, sharex=True)
for t in trainingEnergy:
	axarr[0].plot(t)
axarr[0].set_title('Training set')
axarr[0].set_xlabel('Iterations')
axarr[0].set_ylabel('H')
for v in validationEnergy:
	axarr[1].plot(v)
axarr[1].set_title('Validation set')
axarr[1].set_xlabel('Iterations')
axarr[1].set_ylabel('H')


f.subplots_adjust(hspace=0.3)
plt.show()
