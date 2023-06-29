import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from EducationDataLoader import EducationDataLoader
from sklearn.metrics import r2_score


class StateServer:
	def __init__(self, initialModel: nn.Module, testDataList: list[pd.DataFrame], batchSize) -> None:
		self.globalModel = initialModel
		self.testDataList = testDataList
		self.batchSize = batchSize
		self.allTestData = pd.concat(self.testDataList)
		self.allTestDataLoader = DataLoader(EducationDataLoader(self.allTestData), batch_size=self.batchSize, shuffle=True)

	def aggregate_models(self, models_list: list[nn.Module]):
		globalDict = self.globalModel.state_dict()
		for name, param in globalDict.items():  #name of layer and layer parameters
			globalDict[name] = torch.mean(torch.stack([models.state_dict()[name] for models in models_list]),
										  dim=0)  #averaging all parameters in each layer
		self.globalModel.load_state_dict(globalDict)  #updating global model with new parameters

	def evaluate_model_mae_r2(self):
		testInput = []
		testOutput = []

		#separate input and output in batches
		for batch in self.allTestDataLoader:
			batchInput, batchOutput = batch
			testInput.append(batchInput)
			testOutput.append(batchOutput)

		#concatenate inputs and outputs into one tensor each
		testInput = torch.cat(testInput, dim=0)
		testOutput = torch.cat(testOutput, dim=0)

		#predict values with testing data input
		yPred = self.globalModel(testInput)

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()
		#print("Mean Absolute Error:", mae.item())

		#Convert the tensors to numpy arrays
		npTestOutput = testOutput.detach().numpy()
		npYPred = yPred.detach().numpy()

		#Calculate Coefficient of Determination (r^2)
		r2 = r2_score(npTestOutput, npYPred)
		#print("R^2 Score:", r2)

		return mae, r2

	def get_globalModel(self):
		return self.globalModel
