import pandas as pd
from DataPreprocessing import DataPreprocessing
from DistrictClient import DistrictClient

class DataProvider:
	def __init__(self, dataset: str, numClients: int, dependentVariable: str):
		self.df = None

		#If Dataset is set to SchoolGrades22.xlsx, read the data
		if dataset == "SchoolGrades22.xlsx":
			self.df = pd.read_excel(dataset, sheet_name="SG", skiprows=range(4))
		else:
			assert dataset == "SchoolGrades22.xlsx", "Your dataset is not supported yet!"

		#Make DataPreprocessing object and get condensed data to write to .csv
		self.allData = DataPreprocessing(self.df, dependentVariable)
		self.df = self.allData.get_condensed_data()
		self.df.to_csv("SchoolGrades22 PostProcessed.csv", index=False)

		self.numClients = numClients


	def split_data_to_clients(self) -> list[pd.DataFrame]:
		allClientsList = []

		for i in range(1, self.numClients + 1):
			allClientsList.append(self.allData.get_district_data(i))

		return allClientsList