import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from DataTools.EducationDataLoader import EducationDataLoader
from FLElements.NeuralNetworkNet import NeuralNetworkNet


class StateServer:
	def __init__(self, initialModel: nn.Module, batchSize, testDataList: list[pd.DataFrame], ifPersonalized: bool) -> None:
		self.globalModel = initialModel
		self.batchSize = batchSize

		#testData split by district
		self.testDataList = testDataList

		# self.allTestData = pd.concat(self.testDataList)
		# self.allTestDataLoader = DataLoader(EducationDataLoader(self.allTestData), batch_size=self.batchSize, shuffle=True)

		self.testDataLoaderList = [DataLoader(EducationDataLoader(self.testDataList[i]), batch_size=self.batchSize, shuffle=True) for i in
								   range(len(self.testDataList))]

		if ifPersonalized:
			nodeSizes = []

			for name, param in self.globalModel.named_parameters():
				if 'weight' in name:  # We are interested in weight parameters
					layer_size = param.shape[1]  # The second dimension of weight represents the size of nodes in that layer
					nodeSizes.append(layer_size)

			nodeSizes.append(1)

			self.allPersonalModels = []

			for i in range(len(self.testDataList)):
				self.allPersonalModels.append(NeuralNetworkNet(nodeSizes))

	def aggregate_models(self, localModelsList: list[nn.Module]):
		globalDict = self.globalModel.state_dict()
		for name, param in globalDict.items():  #name of layer and layer parameters
			globalDict[name] = torch.mean(torch.stack([models.state_dict()[name] for models in localModelsList]),
										  dim=0)  #averaging all parameters in each layer
		self.globalModel.load_state_dict(globalDict)  #updating global model with new parameters

	def personalized_aggregate_models(self, localModelsList: list[nn.Module], globalWeightage):
		globalDict = self.globalModel.state_dict()
		listLocalDicts = [model.state_dict() for model in localModelsList]

		listPersonalizedDicts = listLocalDicts[:]

		#print(listPersonalizedDicts[0])

		indexNum = 0

		for localDict in listLocalDicts:
			for key in localDict:
				listPersonalizedDicts[indexNum][key] = globalDict[key] * globalWeightage + listLocalDicts[indexNum][key] * (1 - globalWeightage)

			indexNum += 1

		for i in range(len(self.allPersonalModels)):
			self.allPersonalModels[i].load_state_dict(listPersonalizedDicts[i])

	#print(listPersonalizedDicts[0])

	def evaluate_model_mae_r2(self):
		testInput = torch.empty(0)
		testOutput = torch.empty(0)

		#separate input and output in batches
		for districtDataLoader in self.testDataLoaderList:
			for batchInput, batchOutput in districtDataLoader:
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

	def evaluate_personalized_models_mae_r2(self):
		testOutput = torch.empty(0)
		yPreds = torch.empty(0)

		for i in range(len(self.allPersonalModels)):
			testInput = torch.empty(0)

			#separate input and output in batches
			for batchInput, batchOutput in self.testDataLoaderList[i]:
				testInput = torch.cat((testInput, batchInput), dim=0)
				testOutput = torch.cat((testOutput, batchOutput), dim=0)

			#predict values with testing data input
			yPreds = torch.cat((yPreds, self.allPersonalModels[i](testInput)))

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPreds - testOutput).mean()

		#Calculate Coefficient of Determination (r^2)
		sumOfSquaresOfResiduals = torch.sum(torch.square(yPreds - testOutput))
		sumOfSquaresTotal = torch.sum(torch.square(testOutput - torch.mean(testOutput)))
		r2 = 1 - (sumOfSquaresOfResiduals / sumOfSquaresTotal)

		return mae, r2
