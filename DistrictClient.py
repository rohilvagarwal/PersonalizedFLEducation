import pandas as pd
from EducationDataLoader import EducationDataLoader
from NeuralNetworkNet import NeuralNetworkNet
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score

torch.manual_seed(42)


class DistrictClient:
	def __init__(self, data):
		self.data = data
		self.randomizedData = None
		self.trainingData = None
		self.testingData = None

		self.split_train_and_test()

		self.numEpochs = 500
		self.batch_size = 64

		self.trainDataLoader = DataLoader(EducationDataLoader(self.trainingData), batch_size=self.batch_size, shuffle=True)
		self.testDataLoader = DataLoader(EducationDataLoader(self.testingData), batch_size=self.batch_size, shuffle=True)

	def get_data(self):
		return self.data

	def get_randomized_data(self):
		return self.randomizedData

	def split_train_and_test(self):
		#randomizing order of data
		self.randomizedData = self.data.sample(frac=1, random_state=30).reset_index(drop=True)

		#80% training 20% testing
		amtTraining = int(len(self.randomizedData) * 4 / 5)

		#Training data is 4/5 of total data - will separate into input and output in dataloader
		self.trainingData = self.randomizedData.iloc[:amtTraining, :]
		self.testingData = self.randomizedData.iloc[amtTraining:, :]

	# def get_train_input_and_output(self):
	# 	trainInput = []
	# 	trainOutput = []
	#
	# 	for batch in self.trainDataLoader:
	# 		batchInput, batchOutput = batch
	# 		trainInput.append(batchInput)
	# 		trainOutput.append(batchOutput)
	#
	# 	trainInput = torch.cat(trainInput, dim=0)
	# 	trainOutput = torch.cat(trainOutput, dim=0)
	#
	# 	return trainInput, trainOutput
	#
	# def get_test_input_and_output(self):
	# 	testInput = []
	# 	testOutput = []
	#
	# 	for batch in self.testDataLoader:
	# 		batchInput, batchOutput = batch
	# 		testInput.append(batchInput)
	# 		testOutput.append(batchOutput)
	#
	# 	testInput = torch.cat(testInput, dim=0)
	# 	testOutput = torch.cat(testOutput, dim=0)
	#
	# 	return testInput, testOutput

	def train_neural_network(self):
		# #separating training input and output data
		# trainInput, trainOutput = self.get_train_input_and_output()
		# print(f"Train Input: {trainInput.size()}")
		# print(f"Train Output: {trainOutput.size()}")
		#
		#
		# testInput, testOutput = self.get_test_input_and_output()
		# print(f"Test Input: {testInput.size()}")
		# print(f"Test Output: {testOutput.size()}")

		#dimension sizes
		inputDim = self.data.shape[1] - 1
		hiddenLayer1Dim = 64
		hiddenLayer2Dim = 128
		hiddenLayer3Dim = 128
		hiddenLayer4Dim = 64
		outputDim = 1

		#define neural network and loss/optimizers
		model = NeuralNetworkNet([inputDim, 64, 128, 64, 32, outputDim])
		loss_fn = nn.MSELoss()  # Mean Squared Error loss
		optimizer = optim.Adam(model.parameters(), lr=0.001)

		#forward and backward propagation in batches for numEpochs
		for epoch in range(self.numEpochs):
			for i, (xBatch, yBatch) in enumerate(self.trainDataLoader):
				#xBatch = trainInput[i:i + batchSize]
				yPredToTrain = model(xBatch)
				#yBatch = trainOutput[i:i + batchSize]
				loss = loss_fn(yPredToTrain, yBatch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		testInput = []
		testOutput = []
		for batch in self.testDataLoader:
			batchInput, batchOutput = batch
			testInput.append(batchInput)
			testOutput.append(batchOutput)
		testInput = torch.cat(testInput, dim=0)
		testOutput = torch.cat(testOutput, dim=0)

		yPred = model(testInput)

		print(yPred)
		print(yPred.size())
		print(testOutput)
		print(testOutput.size())

		print()

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()
		print("Mean Absolute Error:", mae.item())

		#Convert the tensors to numpy arrays
		npTestOutput = testOutput.detach().numpy()
		npYPred = yPred.detach().numpy()
		r2 = r2_score(npTestOutput, npYPred)
		print("R^2 Score:", r2)