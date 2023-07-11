from torch import nn
import torch


class NeuralNetworkNet(nn.Module):
	def __init__(self, numNodesPerLayer):
		super(NeuralNetworkNet, self).__init__()

		self.numNodesPerLayer = numNodesPerLayer
		self.linearActions = nn.ModuleList()

		torch.manual_seed(42)
		#adding all layer actions based on the number of nodes per layer
		for x in range(len(numNodesPerLayer) - 1):
			self.linearActions.append(nn.Linear(numNodesPerLayer[x], numNodesPerLayer[x + 1]))

		#defining activation function and dropout layer
		self.activationFunction = nn.ReLU()
		self.dropout = nn.Dropout()

	def forward(self, x):
		#all actions in each forward propagation
		x = self.linearActions[0](x)  #input
		x = self.dropout(x)
		x = self.activationFunction(x)

		if len(self.linearActions) > 2:
			for i in range(1, len(self.linearActions) - 1):
				x = self.linearActions[i](x)
				x = self.activationFunction(x)

		x = self.linearActions[-1](x)  #output
		return x.squeeze()  # Squeeze the tensor to remove the extra dimension
