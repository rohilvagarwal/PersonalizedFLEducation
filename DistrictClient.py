import pandas as pd
from EducationDataLoader import EducationDataLoader
from NeuralNetworkNet import NeuralNetworkNet
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score

class DistrictClient:
	def __init__(self, data):
		self.data = data
		self.randomizedData = None
		self.trainingData = None
		self.testingData = None

		self.split_train_and_test()

		self.trainDataLoader = DataLoader(EducationDataLoader(self.trainingData), batch_size=64, shuffle=True)
		self.testDataLoader = DataLoader(EducationDataLoader(self.testingData), batch_size=64, shuffle=True)

	def get_data(self):
		return self.data

	def get_randomized_data(self):
		return self.randomizedData

	def split_train_and_test(self):
		#randomizing order of data
		self.randomizedData = self.data.sample(frac=1).reset_index(drop=True)
		#self.randomizedData = self.data

		#80% training 20% testing
		amtTraining = int(len(self.randomizedData) * 4 / 5)

		#Training data is 4/5 of total data - will separate into input and output in dataloader
		self.trainingData = self.randomizedData.iloc[:amtTraining, :]
		self.testingData = self.randomizedData.iloc[amtTraining:, :]

	def get_train_input_and_output(self):
		trainInput, trainOutput = next(iter(self.trainDataLoader))
		return trainInput, trainOutput

	def get_test_input_and_output(self):
		testInput, testOutput = next(iter(self.testDataLoader))
		return testInput, testOutput

	def train_neural_network(self):
		#separating training input and output data
		trainInput, trainOutput = self.get_train_input_and_output()
		testInput, testOutput = self.get_test_input_and_output()

		#dimension sizes
		input_dim = 12
		hidden_dim = 144
		output_dim = 1

		#define neural network and loss/optimizers
		model = NeuralNetworkNet(input_dim, hidden_dim, output_dim)
		loss_fn = nn.MSELoss()  # Mean Squared Error loss
		optimizer = optim.Adam(model.parameters(), lr=0.001)

		numEpochs = 500
		batchSize = 10

		#forward and backward propagation in batches for numEpochs
		for epoch in range(numEpochs):
			for i in range(0, len(trainInput), batchSize):
				xBatch = trainInput[i:i + batchSize]
				yPred = model(xBatch)
				yBatch = trainOutput[i:i + batchSize]
				loss = loss_fn(yPred, yBatch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		yPred = model(testInput)

		print(yPred)
		print(testOutput)

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()
		print("Mean Absolute Error:", mae.item())

		#Convert the tensors to numpy arrays
		npTestOutput = testOutput.detach().numpy()
		npYPred = yPred.detach().numpy()
		r2 = r2_score(npTestOutput, npYPred)
		print("R^2 Score:", r2)