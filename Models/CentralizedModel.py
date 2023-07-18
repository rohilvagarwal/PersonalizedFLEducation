import pandas as pd
import torch
from torch import nn
from config import Config
from DataTools.DataProvider import DataProvider, write_evaluation_to_excel, plot_data, write_test_data_to_excel
from FLElements.NeuralNetworkNet import NeuralNetworkNet
import torch.optim as optim
from torch.utils.data import DataLoader
from DataTools.EducationDataLoader import EducationDataLoader

torch.manual_seed(42)

#use gpu if available, otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Current device for training: {device}")

if __name__ == "__main__":
	#Step 1: Set all args
	args = Config().get_args()

	dataset, dependentVariable, numClients, numEpochs = args.dataset, args.dependentVariable, args.numClients, args.aggregatingEpochs
	batchSize, optimizer, learningRate = args.batchSize, args.optimizer, args.learningRate
	hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim = args.hiddenLayer1Dim, args.hiddenLayer2Dim, args.hiddenLayer3Dim, args.hiddenLayer4Dim

	#Step 2: Prepare dataset and split data amongst clients
	dataProvider = DataProvider(dataset, numClients, dependentVariable)
	allData: pd.DataFrame = dataProvider.get_all_data()  #raw data of all districts after preprocessing
	numIndependentVariables = allData.shape[1] - 1

	#Step 3: Build the initial model
	inputDim = numIndependentVariables
	outputDim = 1

	model = NeuralNetworkNet([inputDim, hiddenLayer1Dim, hiddenLayer2Dim, hiddenLayer3Dim, hiddenLayer4Dim, outputDim])
	#initialModel = model.to(device)

	#Step 4: Split Train and Test Data
	randomizedData = allData.sample(frac=1, random_state=30).reset_index(drop=True)
	amtTraining = int(len(randomizedData) * 4 / 5)
	trainingData = randomizedData.iloc[:amtTraining, :]
	testingData = randomizedData.iloc[amtTraining:, :]

	trainDataLoader = DataLoader(EducationDataLoader(trainingData), batch_size=batchSize, shuffle=True)
	testDataLoader = DataLoader(EducationDataLoader(testingData), batch_size=batchSize, shuffle=True)

	#Train Neural Network
	#define loss/optimizer
	loss_fn = nn.MSELoss()  # Mean Squared Error loss

	if optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=learningRate)
	else:
		assert optimizer == "Adam", "This optimizer is not supported"

	maeValues = []
	r2Values = []
	#forward and backward propagation in batches for numEpochs
	for epoch in range(numEpochs):
		for i, (xBatch, yBatch) in enumerate(trainDataLoader):
			yPredToTrain = model(xBatch)
			loss = loss_fn(yPredToTrain, yBatch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		testInput = torch.empty(0)
		testOutput = torch.empty(0)

		#separate input and output in batches
		for batchInput, batchOutput in testDataLoader:
			testInput = torch.cat((testInput, batchInput), dim=0)
			testOutput = torch.cat((testOutput, batchOutput), dim=0)

		#predict values with testing data input
		yPred = model(testInput)

		#Calculate Mean Absolute Error (MAE)
		mae = torch.abs(yPred - testOutput).mean()

		#Calculate Coefficient of Determination (r^2)
		sumOfSquaresOfResiduals = torch.sum(torch.square(yPred - testOutput))
		sumOfSquaresTotal = torch.sum(torch.square(testOutput - torch.mean(testOutput)))
		r2 = 1 - (sumOfSquaresOfResiduals / sumOfSquaresTotal)

		if epoch == numEpochs - 1:
			write_test_data_to_excel(testOutput, yPred, "CentralizedModel")  #evaluate global model

		maeValues.append(mae.item())
		r2Values.append(r2.item())

		print(f"Epoch {epoch + 1}:\nMean Absolute Error: {mae}\nr^2: {r2}")
		print()

	write_evaluation_to_excel(maeValues, r2Values, "CentralizedModel")

	# Plot test accuracy
	plot_data(numEpochs, maeValues, "Mean Absolute Error over Epochs", "Epochs", "Mean Absolute Error", 0, 15)
	plot_data(numEpochs, r2Values, "Coefficient of Determination over Epochs", "Epochs", "Coefficient of Determination", 0, 1)
