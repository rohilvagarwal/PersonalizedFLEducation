import pandas as pd


def get_district_num(row):
	#add Florida district number
	floridaDistrict1Counties = ["CHARLOTTE", "COLLIER", "DESOTO", "GLADES", "HARDEE", "HENDRY", "HIGHLANDS", "LEE", "MANATEE", "OKEECHOBEE",
								"POLK", "SARASOTA"]
	floridaDistrict2Counties = ["ALACHUA", "BAKER", "BRADFORD", "CLAY", "COLUMBIA", "DIXIE", "DUVAL", "GILCHRIST", "HAMILTON", "LAFAYETTE",
								"LEVY", "MADISON", "NASSAU", "PUTNAM", "ST. JOHNS", "SUWANNEE", "TAYLOR", "UNION"]
	floridaDistrict3Counties = ["BAY", "CALHOUN", "ESCAMBIA", "FRANKLIN", "GADSDEN", "GULF", "HOLMES", "JACKSON", "JEFFERSON", "LEON", "LIBERTY",
								"OKALOOSA", "SANTA ROSA", "WAKULLA", "WALTON", "WASHINGTON"]
	floridaDistrict4Counties = ["BROWARD", "INDIAN RIVER", "MARTIN", "PALM BEACH", "ST. LUCIE"]
	floridaDistrict5Counties = ["BREVARD", "FLAGLER", "LAKE", "MARION", "ORANGE", "OSCEOLA", "SEMINOLE", "SUMTER", "VOLUSIA"]
	floridaDistrict6Counties = ["MIAMI-DADE", "MONROE"]
	floridaDistrict7Counties = ["CITRUS", "HERNANDO", "HILLSBOROUGH", "PASCO", "PINELLAS"]

	if row['District Name'] in floridaDistrict1Counties:
		return 1
	elif row['District Name'] in floridaDistrict2Counties:
		return 2
	elif row['District Name'] in floridaDistrict3Counties:
		return 3
	elif row['District Name'] in floridaDistrict4Counties:
		return 4
	elif row['District Name'] in floridaDistrict5Counties:
		return 5
	elif row['District Name'] in floridaDistrict6Counties:
		return 6
	elif row['District Name'] in floridaDistrict7Counties:
		return 7
	else:
		return None


class DataPreprocessing:
	def __init__(self, allData: pd.DataFrame, dependentVariable: str):
		self.allData = allData
		self.condensedData = allData
		self.dependentVariable = dependentVariable #Mathematics Achievement, Mathematics Learning Gains, Mathematics Learning Gains of the Lowest 25%
		self.all_preprocessing()

	def add_district_number(self):
		self.condensedData.insert(0, 'Florida District Number', 0)
		self.condensedData['Florida District Number'] = self.condensedData.apply(get_district_num, axis=1)

	def del_unneeded_columns(self):
		#Get rid of unneeded columns
		self.condensedData = self.condensedData.drop('District Number', axis=1)
		self.condensedData = self.condensedData.drop('District Name', axis=1)
		self.condensedData = self.condensedData.drop('School Number', axis=1)
		self.condensedData = self.condensedData.drop('School Name', axis=1)
		# self.condensedData = self.condensedData.drop('English Language Arts Achievement', axis=1)
		# self.condensedData = self.condensedData.drop('English Language Arts Learning Gains', axis=1)
		# self.condensedData = self.condensedData.drop('English Language Arts Learning Gains of the Lowest 25%', axis=1)
		# self.condensedData = self.condensedData.drop('Science Achievement', axis=1)
		self.condensedData = self.condensedData.drop('Social Studies Achievement', axis=1)
		self.condensedData = self.condensedData.drop('Middle School Acceleration', axis=1)
		self.condensedData = self.condensedData.drop('Graduation Rate 2020-21', axis=1)
		self.condensedData = self.condensedData.drop('College and Career Acceleration 2020-21', axis=1)
		self.condensedData = self.condensedData.drop('Total Points Earned', axis=1)
		self.condensedData = self.condensedData.drop('Total Components', axis=1)
		self.condensedData = self.condensedData.drop('Percent of Total Possible Points', axis=1)
		self.condensedData = self.condensedData.drop('Percent Tested', axis=1)
		self.condensedData = self.condensedData.drop('Optional Grade 2021*', axis=1)
		self.condensedData = self.condensedData.drop('Informational Baseline Grade 2015', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2014', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2013', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2012', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2011', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2010', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2009', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2008', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2007', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2006', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2005', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2004', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2003', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2002', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2001', axis=1)
		self.condensedData = self.condensedData.drop('Grade 2000', axis=1)
		self.condensedData = self.condensedData.drop('Grade 1999', axis=1)
		self.condensedData = self.condensedData.drop('Was the collocated rule used?', axis=1)
		self.condensedData = self.condensedData.drop('Collocated Number', axis=1)
		self.condensedData = self.condensedData.drop('Alternative/ESE Center School', axis=1)

	def del_empty_rows(self):
		#dropping all empty values
		self.condensedData = self.condensedData.dropna(subset=['Florida District Number'])
		self.condensedData = self.condensedData.dropna(subset=['Mathematics Achievement'])
		self.condensedData = self.condensedData.dropna(subset=['Mathematics Learning Gains'])
		self.condensedData = self.condensedData.dropna(subset=['Grade 2022'])
		self.condensedData = self.condensedData.dropna(subset=['Title I'])
		self.condensedData = self.condensedData.dropna(subset=['School Type'])
		self.condensedData = self.condensedData.dropna(subset=['Percent of Minority Students'])
		self.condensedData = self.condensedData.dropna(subset=['Percent of Economically Disadvantaged Students'])
		self.condensedData = self.condensedData.dropna(subset=['Mathematics Learning Gains of the Lowest 25%'])

		self.condensedData = self.condensedData.dropna(subset=['English Language Arts Achievement'])
		self.condensedData = self.condensedData.dropna(subset=['English Language Arts Learning Gains'])
		self.condensedData = self.condensedData.dropna(subset=['English Language Arts Learning Gains of the Lowest 25%'])
		self.condensedData = self.condensedData.dropna(subset=['Science Achievement'])

	def fill_missing_grades(self):
		#substituting latest school grade in for previous years if missing
		self.condensedData['Grade 2019'].fillna(self.condensedData['Grade 2022'], inplace=True)
		self.condensedData['Grade 2018'].fillna(self.condensedData['Grade 2022'], inplace=True)
		self.condensedData['Grade 2017'].fillna(self.condensedData['Grade 2022'], inplace=True)
		self.condensedData['Grade 2016'].fillna(self.condensedData['Grade 2022'], inplace=True)

	def categorical_data_to_num(self):
		#changing categorical data to numbers
		#dictionary to map categories to numerical values
		gradeMapping = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}

		self.condensedData['Grade 2022'] = self.condensedData['Grade 2022'].map(gradeMapping)
		self.condensedData['Grade 2019'] = self.condensedData['Grade 2019'].map(gradeMapping)
		self.condensedData['Grade 2018'] = self.condensedData['Grade 2018'].map(gradeMapping)
		self.condensedData['Grade 2017'] = self.condensedData['Grade 2017'].map(gradeMapping)
		self.condensedData['Grade 2016'] = self.condensedData['Grade 2016'].map(gradeMapping)

		yesNoMapping = {'NO': 0, 'YES': 1}
		self.condensedData['Charter School'] = self.condensedData['Charter School'].map(yesNoMapping)
		self.condensedData['Title I'] = self.condensedData['Title I'].map(yesNoMapping)

	def move_output_to_end(self):
		#move dependent variable to end
		dependentVar = self.dependentVariable
		allColumnsExceptDependent = [col for col in self.condensedData.columns if col != dependentVar]
		newOrder = allColumnsExceptDependent + [dependentVar]
		self.condensedData = self.condensedData[newOrder]

	def all_preprocessing(self):
		self.add_district_number()
		self.del_unneeded_columns()
		self.del_empty_rows()
		self.fill_missing_grades()
		self.categorical_data_to_num()
		self.move_output_to_end()

	def get_district_data(self, districtNum) -> pd.DataFrame:
		return self.condensedData[self.condensedData['Florida District Number'] == districtNum].iloc[:, 1:]

	def get_condensed_data(self) -> pd.DataFrame:
		return self.condensedData
