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
		global_dict = self.globalModel.state_dict()
		for name, param in global_dict.items():  #name is layer name
			global_dict[name] = torch.mean(torch.stack([models.state_dict()[name] for models in models_list]),
										   dim=0)  #taking average of parameters in each layer
		self.globalModel.load_state_dict(global_dict)  #updating global model with new parameters

	def evaluate_model(self):
		testInput = []
		testOutput = []

		for batch in self.allTestDataLoader:
			batchInput, batchOutput = batch
			testInput.append(batchInput)
			testOutput.append(batchOutput)
		testInput = torch.cat(testInput, dim=0)
		testOutput = torch.cat(testOutput, dim=0)

		yPred = self.globalModel(testInput)

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()
		print("Mean Absolute Error:", mae.item())

		#Convert the tensors to numpy arrays
		npTestOutput = testOutput.detach().numpy()
		npYPred = yPred.detach().numpy()
		r2 = r2_score(npTestOutput, npYPred)
		print("R^2 Score:", r2)

	def get_globalModel(self):
		return self.globalModel