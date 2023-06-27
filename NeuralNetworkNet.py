import torch
from torch import nn
import torch.nn.functional as F


class NeuralNetworkNet(nn.Module):
	def __init__(self, dim_in, dim_hidden, dim_out):
		super(NeuralNetworkNet, self).__init__()
		self.layer_input = nn.Linear(dim_in, dim_hidden)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout()
		self.layer_hidden = nn.Linear(dim_hidden, dim_hidden)
		self.output = nn.Linear(dim_hidden, dim_out)
		# self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		#x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
		x = self.layer_input(x)
		x = self.dropout(x)
		x = self.relu(x)
		x = self.layer_hidden(x)
		x = self.relu(x)
		x = self.output(x)
		#x = self.softmax(x)
		return x.squeeze()  # Squeeze the tensor to remove the extra dimension
