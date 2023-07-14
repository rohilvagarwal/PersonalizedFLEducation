import pandas as pd
import torch
from torch import nn
from config import Config
from DataTools.DataProvider import DataProvider, write_results_to_excel, plot_data
from FLElements.DistrictClient import DistrictClient
from FLElements.NeuralNetworkNet import NeuralNetworkNet
from FLElements.StateServer import StateServer

#use gpu if available, otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device for training: {device}")

if __name__ == "__main__":
	#Step 1: Set all args
	args = Config().get_args()

	dataset, dependentVariable, numClients, aggregatingEpochs = args.dataset, args.dependentVariable, args.numClients, args.aggregatingEpochs
	localEpochs, batchSize, optimizer, learningRate = args.localEpochs, args.batchSize, args.optimizer, args.learningRate

	#Step 2: Prepare dataset and split data amongst clients
	dataProvider = DataProvider(dataset, numClients, dependentVariable)
	clientsData: list[pd.DataFrame] = dataProvider.split_data_to_clients()  #list of raw data of each district after preprocessing
	numIndependentVariables = clientsData[0].shape[1] - 1

	#Step 3: Build the initial model
	inputDim = numIndependentVariables
	hiddenLayer1Dim = 64
	hiddenLayer2Dim = 128
	hiddenLayer3Dim = 64
	hiddenLayer4Dim = 32
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

	server = StateServer(globalInitialModel, batchSize, allTestingDataList, False)

	#Step 6: Build the training procedure
	maeValues = []
	r2Values = []

	#repeat for aggregatingEpochs
	for epoch in range(aggregatingEpochs):
		allLocalModels: list[nn.Module] = []

		for client in clientsList:
			#train local model and append to allLocalModels list
			client.train_neural_network()
			allLocalModels.append(client.get_model())

		server.aggregate_models(allLocalModels)  #aggregates all local models
		globalModel = server.get_globalModel()  #gets global model from aggregated local models

		mae, r2 = server.evaluate_model_mae_r2()  #evaluate global model

		maeValues.append(mae)
		r2Values.append(r2)

		print(f"Epoch {epoch + 1}:\nMean Absolute Error: {mae}\nr^2: {r2}")
		print()

		#set local model to previous global model
		for client in clientsList:
			client.grab_global_model(globalModel)

	#write_results_to_excel(maeValues, r2Values, "FLModel")

	data = {'epochs': list(range(1, len(maeValues) + 1)),
			'MAE': maeValues,
			'R2': r2Values}
	df = pd.DataFrame(data)
	writer = pd.ExcelWriter('../Data/results.xlsx', engine='xlsxwriter')
	df.to_excel(writer, sheet_name="FLModel", index=False)
	writer.close()

	#Plot test accuracy
	plot_data(aggregatingEpochs, maeValues, "Mean Absolute Error over Epochs", "Epochs", "Mean Absolute Error", 0, 15)
	plot_data(aggregatingEpochs, r2Values, "Coefficient of Determination over Epochs", "Epochs", "Coefficient of Determination", 0, 1)