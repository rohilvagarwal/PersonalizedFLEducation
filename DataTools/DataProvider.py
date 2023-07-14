import pandas as pd
from DataTools.DataPreprocessing import DataPreprocessing
import os
import matplotlib.pyplot as plt
import numpy as np


def write_results_to_excel(listMAE, listR2, sheetName):
	data = {'Epochs': list(range(1, len(listMAE) + 1)),
			'MAE': listMAE,
			'R2': listR2}
	df = pd.DataFrame(data)
	writer = pd.ExcelWriter('../Data/results.xlsx', engine='xlsxwriter')
	df.to_excel(writer, sheet_name=sheetName, index=False)
	writer.close()


def plot_data(aggregatingEpochs, values, title, xLabel, yLabel, yMin, yMax):
	plt.plot(np.arange(1, aggregatingEpochs + 1), values)
	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.ylim(yMin, yMax)
	plt.show()


class DataProvider:
	def __init__(self, dataset: str, numClients: int, dependentVariable: str):
		self.df = None

		#If Dataset is set to SchoolGrades22.xlsx, read the data
		# Get the path to the current file
		current_dir = os.path.dirname(os.path.abspath(__file__))

		# Construct the path to the Excel file
		excel_file_path = os.path.join(current_dir, "../Data/SchoolGrades22.xlsx")

		if dataset == "SchoolGrades22.xlsx":
			self.df = pd.read_excel(excel_file_path, sheet_name="SG", skiprows=range(4))
		else:
			assert dataset == "SchoolGrades22.xlsx", "Your dataset is not supported yet!"

		#Make DataPreprocessing object and get condensed data to write to .csv
		self.allData = DataPreprocessing(self.df, dependentVariable)
		self.df = self.allData.get_condensed_data()
		self.df.to_csv("../data/SchoolGrades22 PostProcessed.csv", index=False)

		if numClients == 7:
			self.numClients = numClients
		else:
			assert numClients == 7, "Your number of clients is not supported yet!"

	def get_all_data(self) -> pd.DataFrame:
		return self.df

	def split_data_to_clients(self) -> list[pd.DataFrame]:
		#return list with data from each client
		allClientsList = []

		for i in range(1, self.numClients + 1):
			allClientsList.append(self.allData.get_district_data(i))

		return allClientsList
