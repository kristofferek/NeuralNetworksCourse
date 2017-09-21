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

        # Weights
        self.wj = (0.2 - (-0.2)) * np.random.random_sample((2,4)) + (-0.2)
        self.wi =  (0.2 - (-0.2)) * np.random.random_sample((4,1)) + (-0.2)
        # Thresholds
        self.outputTS = (1 - (-1)) * np.random.random_sample() + (-1)
        self.hiddenTS = (1 - (-1)) * np.random.random_sample((4,1)) + (-1)

    def actFunc(self, b, deriv=False):
        if (deriv == True):
        	return self.beta*(1-np.tanh(self.beta*b)**2)
        return np.tanh(self.beta*b)

    def trainNetwork(self):
        rand = randint(0,len(self.inputData)-1)
        # Forward propagation
        inputs = (self.inputData[rand])[np.newaxis].T
        bj = np.dot(np.transpose(self.wj),inputs) - self.hiddenTS
        V = self.actFunc(bj)

        bi = np.dot(np.transpose(V), self.wi) - self.outputTS
        output = self.actFunc(bi)

        #Backwards propagation
        outputError = self.targetOutput[rand] - output
        di = outputError * self.actFunc(bi, True)

        VError = np.dot(di,np.transpose(self.wi))
        dj = np.transpose(VError) * self.actFunc(bj,True)

        #Updates
        self.wi = np.add(self.wi, np.dot(V, self.learnR*di))
        self.outputTS = np.add(self.outputTS, -(self.learnR*di))
        self.wj = np.add(self.wj, np.dot(inputs, self.learnR*np.transpose(dj)))
        self.hiddenTS = np.add(self.hiddenTS, -(self.learnR*dj))

    def getOutputForAllP(self, input, target):
        # Forward propagation
        bj = np.dot(input, self.wj) - np.transpose(self.hiddenTS)
        v = self.actFunc(bj)

        bi = np.dot(v, self.wi) - self.outputTS
        return self.actFunc(bi)


    def energyFunc(self, input, target):
        output = self.getOutputForAllP(input, target)

        sum = np.sum(np.square(target[np.newaxis].T - output))
        return sum/2

    def classError(self, input, target):
        output = self.getOutputForAllP(input, target)

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
    net = Backpropagation()
    net.readData()
    net.initValues(learnR, beta)

    # Learning iterations
    for i in range(1,1000000):
        net.trainNetwork()
        t.append(net.energyFunc(net.inputData, net.targetOutput))
        v.append(net.energyFunc(net.V_inputData, net.V_targetOutput))
        if i % 100000 == 0:
            print(i)
    
    print('Experiment: ', experiment+1)
    trainingEnergy.append(t)
    validationEnergy.append(v)

    # Training classification error
    cErr = net.classError(net.inputData, net.targetOutput)
    C_T.append(cErr)

    # Validation classification error
    cErr = net.classError(net.V_inputData, net.V_targetOutput)
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