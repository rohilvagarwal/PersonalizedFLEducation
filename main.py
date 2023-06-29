import pandas as pd
from DataPreprocessing import DataPreprocessing
from DistrictClient import DistrictClient
from config import Config
from DataProvider import DataProvider
from NeuralNetworkNet import NeuralNetworkNet
from StateServer import StateServer
from torch.utils.data import DataLoader
from EducationDataLoader import EducationDataLoader
import argparse
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device for training: {device}")

if __name__ == "__main__":
	#Step 1: Set all args
	args = Config().get_args()

	dataset, dependentVariable, numClients, aggregatingEpochs = args.dataset, args.dependentVariable, args.numClients, args.aggregatingEpochs
	localEpochs, batchSize, optimizer, learningRate = args.localEpochs, args.batchSize, args.optimizer, args.learningRate

	#Step 2: Prepare dataset and split data amongst clients
	dataProvider = DataProvider(dataset, numClients, dependentVariable)
	clientsData: list[pd.DataFrame] = dataProvider.split_data_to_clients()
	numIndependentVariables = clientsData[0].shape[1] - 1

	#Step 3: Build the initial model
	inputDim = numIndependentVariables
	hiddenLayer1Dim = 64
	hiddenLayer2Dim = 128
	hiddenLayer3Dim = 64
	hiddenLayer4Dim = 32
	outputDim = 1

	model = NeuralNetworkNet([inputDim, hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim, outputDim])
	initialModel = model.to(device)

	#Step 4: Initialize Clients
	clients_list: list[DistrictClient] = [DistrictClient(clientsData[i], initialModel, localEpochs, batchSize, optimizer, learningRate) for i in range(numClients)]

	#Step 5: Initialize the server
	#get all testing data in list
	allTestingDataList: list[pd.DataFrame] = [clients_list[i].get_testingData() for i in range(numClients)]

	server = StateServer(initialModel, allTestingDataList, batchSize)

	#Step 6: Build the training procedure
	allLocalModels: list[nn.Module] = []

	for i in range(aggregatingEpochs):
		for client in clients_list:
			client.train_neural_network()
			allLocalModels.append(client.get_model())

		server.aggregate_models(allLocalModels)  #aggregates model
		global_model = server.get_globalModel() #gets global model

		evaluate_result = server.evaluate_model()  #test model

		for client in clients_list:
			client.grab_global_model(global_model)

	#district1.train_neural_network()
	#allDataNoDistrict = DistrictClient(df.iloc[:, 1:]).train_neural_network()
