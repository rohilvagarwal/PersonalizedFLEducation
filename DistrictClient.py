import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from EducationDataLoader import EducationDataLoader

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

	def train_neural_network(self):
		#define loss/optimizer
		loss_fn = nn.MSELoss()  # Mean Squared Error loss
		optimizer = None

		if self.optimizer == "Adam":
			optimizer = optim.Adam(self.localModel.parameters(), lr=self.learningRate)
		else:
			assert self.optimizer == "Adam", "This optimizer is not supported"

		#forward and backward propagation in batches for numEpochs
		for epoch in range(self.numEpochs):
			for i, (xBatch, yBatch) in enumerate(self.trainDataLoader):
				yPredToTrain = self.localModel(xBatch)
				loss = loss_fn(yPredToTrain, yBatch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	def grab_global_model(self, globalModel: nn.Module):
		self.localModel.load_state_dict(globalModel.state_dict())