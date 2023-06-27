import pandas as pd
from EducationDataLoader import EducationDataLoader
from torch.utils.data import DataLoader

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