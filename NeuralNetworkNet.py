import torch
from torch import nn
import torch.nn.functional as F


class NeuralNetworkNet(nn.Module):
	def __init__(self, numNodesPerLayer):
		super(NeuralNetworkNet, self).__init__()

		self.numNodesPerLayer = numNodesPerLayer
		self.linearActions = nn.ModuleList()

		#self.linearActions = []

		for x in range(len(numNodesPerLayer) - 1):
			self.linearActions.append(nn.Linear(numNodesPerLayer[x], numNodesPerLayer[x + 1]))

		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()

	def forward(self, x):
		#x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.linearActions[0](x) #input
		x = self.dropout(x)
		x = self.relu(x)

		if len(self.linearActions) > 2:
			for i in range(1, len(self.linearActions) - 1):
				x = self.linearActions[i](x)
				x = self.relu(x)

		x = self.linearActions[-1](x) #output
		return x.squeeze()  # Squeeze the tensor to remove the extra dimension
