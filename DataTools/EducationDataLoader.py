import torch
from torch.utils.data import Dataset

class EducationDataLoader(Dataset):
	def __init__(self, df):
		self.df = df

		self.dfInput = df.iloc[:, :-1]
		self.dfOutput = df.iloc[:, -1]

		self.dfInputTensor = torch.tensor(self.dfInput.values, dtype=torch.float)
		self.dfOutputTensor = torch.tensor(self.dfOutput.values, dtype=torch.float)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx): #used in indexing
		return self.dfInputTensor[idx], self.dfOutputTensor[idx]