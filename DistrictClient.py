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
	def __init__(self, data, initialModel: nn.Module, localEpochs, batchSize, optimizer: str, learningRate):
		self.data = data
		self.randomizedData = None
		self.trainingData = None
		self.testingData = None

		self.split_train_and_test()

		self.localModel = initialModel
		self.numEpochs = localEpochs
		self.batchSize = batchSize

		self.optimizer = optimizer
		self.learningRate = learningRate

		self.trainDataLoader = DataLoader(EducationDataLoader(self.trainingData), batch_size=self.batchSize, shuffle=True)
		self.testDataLoader = DataLoader(EducationDataLoader(self.testingData), batch_size=self.batchSize, shuffle=True)

	def get_data(self) -> pd.DataFrame:
		return self.data

	def get_randomized_data(self):
		return self.randomizedData

	def get_testingData(self) -> pd.DataFrame:
		return self.testingData

	def get_model(self):
		return self.localModel

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

		# #dimension sizes
		# inputDim = self.data.shape[1] - 1
		# hiddenLayer1Dim = 64
		# hiddenLayer2Dim = 128
		# hiddenLayer3Dim = 128
		# hiddenLayer4Dim = 64
		# outputDim = 1

		#define neural network and loss/optimizers
		loss_fn = nn.MSELoss()  # Mean Squared Error loss

		optimizer = None
		if self.optimizer == "Adam":
			optimizer = optim.Adam(self.localModel.parameters(), lr=self.learningRate)
		else:
			assert self.optimizer == "Adam", "This optimizer is not supported"


		#forward and backward propagation in batches for numEpochs
		for epoch in range(self.numEpochs):
			for i, (xBatch, yBatch) in enumerate(self.trainDataLoader):
				#xBatch = trainInput[i:i + batchSize]
				yPredToTrain = self.localModel(xBatch)
				#yBatch = trainOutput[i:i + batchSize]
				loss = loss_fn(yPredToTrain, yBatch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		# testInput = []
		# testOutput = []
		# for batch in self.testDataLoader:
		# 	batchInput, batchOutput = batch
		# 	testInput.append(batchInput)
		# 	testOutput.append(batchOutput)
		# testInput = torch.cat(testInput, dim=0)
		# testOutput = torch.cat(testOutput, dim=0)
		#
		# yPred = self.model(testInput)
		#
		# print(yPred)
		# print(yPred.size())
		# print(testOutput)
		# print(testOutput.size())
		#
		# print()
		#
		# #Calculate Mean Absolute Error (MAE)
		# mae = torch.abs(yPred - testOutput).mean()
		# print("Mean Absolute Error:", mae.item())
		#
		# #Convert the tensors to numpy arrays
		# npTestOutput = testOutput.detach().numpy()
		# npYPred = yPred.detach().numpy()
		# r2 = r2_score(npTestOutput, npYPred)
		# print("R^2 Score:", r2)

	def grab_global_model(self, globalModel: nn.Module):
		self.localModel.load_state_dict(globalModel.state_dict())