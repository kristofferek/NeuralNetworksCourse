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
        #print('inputData ',self.inputData.shape)
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
        #print('V_inputData ', self.V_inputData.shape)
        self.V_targetOutput = np.asarray(self.V_targetOutput)

    def initValues(self, learnR, beta):
        self.learnR = learnR
        self.beta = beta

        # 2x1 matrix, random values [-0.2, 0.2]
        self.smallWeights = (0.2 - (-0.2)) * np.random.random_sample((2,4)) + (-0.2)
        #print('smallWeights ', self.smallWeights.shape)

        self.bigWeights =  (0.2 - (-0.2)) * np.random.random_sample((4,1)) + (-0.2)
        # single random value [-1, 1]
        self.outputTS = (1 - (-1)) * np.random.random_sample() + (-1)

        self.hiddenTS = (1 - (-1)) * np.random.random_sample((1,4)) + (-1)
        #print('hiddenTS ',self.hiddenTS.shape)

    def actFunc(self, b, deriv=False):
        if (deriv == True):
        	return self.beta*(1-np.tanh(self.beta*b)**2)
        return np.tanh(self.beta*b)

    def trainNetwork(self):
        #
        #HIDDEN
        #
        rand = randint(0,len(self.inputData)-1)

        # Forward propagation
        inputs = (self.inputData[rand])[np.newaxis].T
        bj = np.sum(np.dot(np.transpose(self.smallWeights),inputs)) - self.hiddenTS
        #print('bj',bj.shape)
        hidden = self.actFunc(bj)
        #print('V ',hidden.shape)

        bi = np.sum(np.dot(hidden,self.bigWeights)) - self.outputTS
        #print('bi', bi.shape)
        output = self.actFunc(bi)
        #print('output ', output.shape)


        #Backwards propagation
        # Output Error
        outputError = self.targetOutput[rand] - output
        #print('outputError ', outputError.shape)

        # Delta output
        outputDelta = outputError * self.actFunc(output, True)
        outputDelta = self.learnR * outputDelta
        #print('outputDelta ',outputDelta.shape)

        # Hidden Error
        hiddenError = np.dot(outputDelta,np.transpose(self.bigWeights))
        #print('hiddenError ', hiddenError.shape)

        # Delta Hidden
        hiddenDelta = hiddenError * self.actFunc(hidden,True)
        hiddenDelta = self.learnR * hiddenDelta
        #print('hiddenDelta ', hiddenDelta)

        #Updates
        self.bigWeights = np.add(self.bigWeights,np.dot(np.transpose(hidden), outputDelta))
        self.outputTS = np.add(self.outputTS, -outputDelta)
        self.smallWeights = np.add(self.smallWeights,np.dot(inputs, hiddenDelta))
        self.hiddenTS = np.add(self.hiddenTS, -hiddenDelta)
        #print('outputTs ', self.outputTS.shape)
        #print('hiddenTs ', self.hiddenTS.shape)

    def energyFunc(self, input, target):
        # Forward propagation
        bj = np.dot(input, self.smallWeights) - self.hiddenTS
        v = self.actFunc(bj)
        bi = np.dot(v, self.bigWeights) - self.outputTS
        output = self.actFunc(bi)
        #print('bigOutput ', output.shape)
        sum = np.sum(np.square(target[np.newaxis].T - output))
        return sum/2

learnR = 0.02
beta = 0.5

trainingEnergy = []
validationEnergy = []
back = Backpropagation()
back.readData()
back.initValues(learnR, beta)
start = time.time()
for x in range(1,100000):
	back.trainNetwork()
	trainingEnergy.append(back.energyFunc(back.inputData, back.targetOutput))
	validationEnergy.append(back.energyFunc(back.V_inputData, back.V_targetOutput))
	if x % 10000 == 0:
		print(x)
		end = time.time()
		print(end - start, ' seconds since begining')
plt.subplot(2, 1, 1)
plt.plot(trainingEnergy)
plt.subplot(2, 1, 2)
plt.plot(validationEnergy, 'r-')
plt.show()
