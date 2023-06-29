import pandas as pd
from DataPreprocessing import DataPreprocessing
from DistrictClient import DistrictClient
from NeuralNetworkNet import NeuralNetworkNet
from torch.utils.data import DataLoader
from EducationDataLoader import EducationDataLoader

df = pd.read_excel('SchoolGrades22.xlsx', sheet_name="SG", skiprows=range(4))

allData = DataPreprocessing(df, "Mathematics Achievement")
df = allData.get_condensed_data()

district1 = DistrictClient(allData.get_district_data(1))
district2 = DistrictClient(allData.get_district_data(2))
district3 = DistrictClient(allData.get_district_data(3))
district4 = DistrictClient(allData.get_district_data(4))
district5 = DistrictClient(allData.get_district_data(5))
district6 = DistrictClient(allData.get_district_data(6))
district7 = DistrictClient(allData.get_district_data(7))

allDistrictDF = [district1, district2, district3, district4, district5, district6, district7]

#district1.train_neural_network()
allDataNoDistrict = DistrictClient(df.iloc[:, 1:]).train_neural_network()

df.to_csv("SchoolGrades22 PostProcessed.csv", index=False)