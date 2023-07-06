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
		testInput = torch.empty(0)
		testOutput = torch.empty(0)

		#separate input and output in batches
		for batchInput, batchOutput in self.allTestDataLoader:
			testInput = torch.cat((testInput, batchInput), dim=0)
			testOutput = torch.cat((testOutput, batchOutput), dim=0)

		#predict values with testing data input
		yPred = self.globalModel(testInput)

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()

		#Calculate Coefficient of Determination (r^2)
		sumOfSquaresOfResiduals = torch.sum(torch.square(yPred - testOutput))
		sumOfSquaresTotal = torch.sum(torch.square(testOutput - torch.mean(testOutput)))
		r2 = 1 - (sumOfSquaresOfResiduals / sumOfSquaresTotal)

		return mae, r2

	def get_globalModel(self):
		return self.globalModel
