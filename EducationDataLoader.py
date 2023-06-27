import pandas as pd
import torch
from torch.utils.data import Dataset

class EducationDataLoader(Dataset):
	def __init__(self, df):
		self.df = df

		self.dfInput = df.iloc[:, :-1]
		self.dfOutput = df.iloc[:, -1]

		self.dfInputTensor = torch.tensor(self.dfInput.values)
		self.dfOutputTensor = torch.tensor(self.dfOutput.values)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx): #used in indexing
		return self.dfInputTensor[idx], self.dfOutputTensor[idx]