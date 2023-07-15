import argparse


class Config:
	def __init__(self) -> None:
		self.args = self.get_args()

	def get_args(self) -> argparse.Namespace:
		parser = argparse.ArgumentParser(description="Federated Learning in Education")

		#training settings
		parser.add_argument("--dataset", type=str, default="SchoolGrades22.xlsx", help="The dataset that is going to be trained")
		parser.add_argument("--dependentVariable", type=str, default="Mathematics Achievement", help="The value that will be predicted")
		parser.add_argument("--numClients", type=int, default=7, help="Number of Florida Districts/Clients participating in Federated Learning")
		parser.add_argument("--aggregatingEpochs", type=int, default=100, help="Total aggregating epochs")
		parser.add_argument("--globalWeightage", type=float, default=0.7, help="Weightage given to global model")

		#model settings
		parser.add_argument("--localEpochs", type=int, default=3, help="Total local training epochs")
		parser.add_argument("--batchSize", type=int, default=128, help="Batch size each step")
		parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer on the local side")
		parser.add_argument("--learningRate", type=float, default=0.001, help="Learning rate on the local side")

		#hidden layers
		parser.add_argument("--hiddenLayer1Dim", type=int, default=32, help="Number of nodes in hidden layer 1")
		parser.add_argument("--hiddenLayer2Dim", type=int, default=64, help="Number of nodes in hidden layer 2")
		parser.add_argument("--hiddenLayer3Dim", type=int, default=32, help="Number of nodes in hidden layer 3")
		parser.add_argument("--hiddenLayer4Dim", type=int, default=16, help="Number of nodes in hidden layer 4")

		args = parser.parse_args()
		return args
