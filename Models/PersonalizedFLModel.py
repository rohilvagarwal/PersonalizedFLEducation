import pandas as pd
import torch
from torch import nn
from config import Config
from DataTools.DataProvider import DataProvider, write_evaluation_to_excel, plot_data
from FLElements.DistrictClient import DistrictClient
from FLElements.NeuralNetworkNet import NeuralNetworkNet
from FLElements.StateServer import StateServer

#use gpu if available, otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device for training: {device}")

if __name__ == "__main__":
	#Step 1: Set all args
	args = Config().get_args()

	dataset, dependentVariable, numClients, aggregatingEpochs, globalWeightage = args.dataset, args.dependentVariable, args.numClients, args.aggregatingEpochs, args.globalWeightage
	localEpochs, batchSize, optimizer, learningRate = args.localEpochs, args.batchSize, args.optimizer, args.learningRate
	hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim = args.hiddenLayer1Dim, args.hiddenLayer2Dim, args.hiddenLayer3Dim, args.hiddenLayer4Dim

	#Step 2: Prepare dataset and split data amongst clients
	dataProvider = DataProvider(dataset, numClients, dependentVariable)
	clientsData: list[pd.DataFrame] = dataProvider.split_data_to_clients()  #list of raw data of each district after preprocessing
	numIndependentVariables = clientsData[0].shape[1] - 1

	#Step 3: Build the initial model
	inputDim = numIndependentVariables
	outputDim = 1

	clientsList: list[DistrictClient] = []

	#Step 4: Initialize Clients

	for i in range(numClients):
		model = NeuralNetworkNet([inputDim, hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim, outputDim])
		initialModel = model.to(device)

		clientsList.append(DistrictClient(clientsData[i], initialModel, localEpochs, batchSize, optimizer, learningRate))

	#Step 5: Initialize the server
	#get all testing data in list
	allTestingDataList: list[pd.DataFrame] = [clientsList[i].get_testingData() for i in range(numClients)]

	globalModel = NeuralNetworkNet([inputDim, hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim, outputDim])
	globalInitialModel = globalModel.to(device)
	server = StateServer(globalInitialModel, batchSize, allTestingDataList, True)

	#Step 6: Build the training procedure
	maeValues = []
	r2Values = []

	#repeat for aggregatingEpochs
	for epoch in range(aggregatingEpochs):
		allLocalModels: list[nn.Module] = []

		# for client in clientsList:
		# 	print(client.get_model().state_dict())

		for client in clientsList:
			#train local model and append to allLocalModels list
			#print(client.get_model().state_dict())
			client.train_neural_network()
			#print(client.get_model().state_dict())
			allLocalModels.append(client.get_model())

		server.aggregate_models(allLocalModels)  #aggregates all local models
		globalModel = server.get_globalModel()  #gets global model from aggregated local models

		server.create_personalized_models(allLocalModels, globalWeightage)

		if epoch == aggregatingEpochs - 1:
			mae, r2 = server.evaluate_personalized_models_mae_r2(True)  #evaluate global model
		else:
			mae, r2 = server.evaluate_personalized_models_mae_r2(False)

		maeValues.append(mae)
		r2Values.append(r2)

		print(f"Epoch {epoch + 1}:\nMean Absolute Error: {mae}\nr^2: {r2}")
		print()

		#set local model to previous global model
		for client in clientsList:
			client.grab_global_model(globalModel)

	write_evaluation_to_excel(maeValues, r2Values, "PersonalizedFLModel")

	# Plot test accuracy
	plot_data(aggregatingEpochs, maeValues, "Mean Absolute Error over Epochs", "Epochs", "Mean Absolute Error", 0, 15)
	plot_data(aggregatingEpochs, r2Values, "Coefficient of Determination over Epochs", "Epochs", "Coefficient of Determination", 0, 1)
