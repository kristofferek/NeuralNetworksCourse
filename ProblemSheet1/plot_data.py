import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import math

class PlotData:
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

data = PlotData()
data.readData()

posX = []
posY = []
negX = []
negY = []
for i in range(0,300):
	if data.targetOutput[i] == 1:
		posX.append(data.inputData[i][np.newaxis][0, 0])
		posY.append(data.inputData[i][np.newaxis][0, 1])
	else:
		negX.append(data.inputData[i][np.newaxis][0, 0])
		negY.append(data.inputData[i][np.newaxis][0, 1])
for i in range(0,200):
	if data.V_targetOutput[i] == 1:
		posX.append(data.V_inputData[i][np.newaxis][0, 0])
		posY.append(data.V_inputData[i][np.newaxis][0, 1])
	else:
		negX.append(data.V_inputData[i][np.newaxis][0, 0])
		negY.append(data.V_inputData[i][np.newaxis][0, 1])

plt.plot(posX,posY, '.')
plt.plot(negX,negY, '.')
plt.show()