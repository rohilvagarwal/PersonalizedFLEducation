import pandas as pd
from Net import Net

input_dim = 8
hidden_dim = 64
output_dim = 1

model = Net(input_dim, hidden_dim, output_dim)

df = pd.read_excel('SchoolGrades22.xlsx', sheet_name="SG", skiprows=range(4))

#add Florida district number
floridaDistrict1Counties = ["CHARLOTTE", "COLLIER", "DESOTO", "GLADES", "HARDEE", "HENDRY", "HIGHLANDS", "LEE", "MANATEE", "OKEECHOBEE", "POLK", "SARASOTA"]
floridaDistrict2Counties = ["ALACHUA", "BAKER", "BRADFORD", "CLAY", "COLUMBIA", "DIXIE", "DUVAL", "GILCHRIST", "HAMILTON", "LAFAYETTE", "LEVY", "MADISON", "NASSAU", "PUTNAM", "ST. JOHNS", "SUWANNEE", "TAYLOR", "UNION"]
floridaDistrict3Counties = ["BAY", "CALHOUN", "ESCAMBIA", "FRANKLIN", "GADSDEN", "GULF", "HOLMES", "JACKSON", "JEFFERSON", "LEON", "LIBERTY", "OKALOOSA", "SANTA ROSA", "WAKULLA", "WALTON", "WASHINGTON"]
floridaDistrict4Counties = ["BROWARD", "INDIAN RIVER", "MARTIN", "PALM BEACH", "ST. LUCIE"]
floridaDistrict5Counties = ["BREVARD", "FLAGLER", "LAKE", "MARION", "ORANGE", "OSCEOLA", "SEMINOLE", "SUMTER", "VOLUSIA"]
floridaDistrict6Counties = ["MIAMI-DADE", "MONROE"]
floridaDistrict7Counties = ["CITRUS", "HERNANDO", "HILLSBOROUGH", "PASCO", "PINELLAS"]

def get_district_num(row):
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

df.insert(0, 'Florida District Number', 0)
df['Florida District Number'] = df.apply(get_district_num, axis=1)

#Get rid of unneeded columns
df = df.drop('District Number', axis=1)
df = df.drop('District Name', axis=1)
df = df.drop('School Number', axis=1)
df = df.drop('School Name', axis=1)
df = df.drop('English Language Arts Achievement', axis=1)
df = df.drop('English Language Arts Learning Gains', axis=1)
df = df.drop('English Language Arts Learning Gains of the Lowest 25%', axis=1)
df = df.drop('Science Achievement', axis=1)
df = df.drop('Social Studies Achievement', axis=1)
df = df.drop('Middle School Acceleration', axis=1)
df = df.drop('Graduation Rate 2020-21', axis=1)
df = df.drop('College and Career Acceleration 2020-21', axis=1)
df = df.drop('Total Points Earned', axis=1)
df = df.drop('Total Components', axis=1)
df = df.drop('Percent of Total Possible Points', axis=1)
df = df.drop('Percent Tested', axis=1)
df = df.drop('Optional Grade 2021*', axis=1)
df = df.drop('Informational Baseline Grade 2015', axis=1)
df = df.drop('Grade 2014', axis=1)
df = df.drop('Grade 2013', axis=1)
df = df.drop('Grade 2012', axis=1)
df = df.drop('Grade 2011', axis=1)
df = df.drop('Grade 2010', axis=1)
df = df.drop('Grade 2009', axis=1)
df = df.drop('Grade 2008', axis=1)
df = df.drop('Grade 2007', axis=1)
df = df.drop('Grade 2006', axis=1)
df = df.drop('Grade 2005', axis=1)
df = df.drop('Grade 2004', axis=1)
df = df.drop('Grade 2003', axis=1)
df = df.drop('Grade 2002', axis=1)
df = df.drop('Grade 2001', axis=1)
df = df.drop('Grade 2000', axis=1)
df = df.drop('Grade 1999', axis=1)
df = df.drop('Was the collocated rule used?', axis=1)
df = df.drop('Collocated Number', axis=1)
df = df.drop('Alternative/ESE Center School', axis=1)

#dropping all empty values
df = df.dropna(subset=['Florida District Number'])
df = df.dropna(subset=['Mathematics Achievement'])
df = df.dropna(subset=['Mathematics Learning Gains'])
df = df.dropna(subset=['Grade 2022'])
df = df.dropna(subset=['Title I'])
df = df.dropna(subset=['School Type'])
df = df.dropna(subset=['Percent of Minority Students'])
df = df.dropna(subset=['Percent of Economically Disadvantaged Students'])
df = df.dropna(subset=['Mathematics Learning Gains of the Lowest 25%'])

#substituting latest school grade in for previous years if missing
df['Grade 2019'].fillna(df['Grade 2022'], inplace=True)
df['Grade 2018'].fillna(df['Grade 2022'], inplace=True)
df['Grade 2017'].fillna(df['Grade 2022'], inplace=True)
df['Grade 2016'].fillna(df['Grade 2022'], inplace=True)

#changing categorical data to numbers
#dictionary to map categories to numerical values
gradeMapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5}

df['Grade 2022'] = df['Grade 2022'].map(gradeMapping)
df['Grade 2019'] = df['Grade 2019'].map(gradeMapping)
df['Grade 2018'] = df['Grade 2018'].map(gradeMapping)
df['Grade 2017'] = df['Grade 2017'].map(gradeMapping)
df['Grade 2016'] = df['Grade 2016'].map(gradeMapping)

yesNoMapping = {'NO': 0, 'YES': 1}
df['Charter School'] = df['Charter School'].map(yesNoMapping)
df['Title I'] = df['Title I'].map(yesNoMapping)

df.to_csv("SchoolGrades22 PostProcessed.csv", index=False)

#Split dataframe into separate districts
dfDistrict1 = df[df['Florida District Number'] == 1]
dfDistrict2 = df[df['Florida District Number'] == 2]
dfDistrict3 = df[df['Florida District Number'] == 3]
dfDistrict4 = df[df['Florida District Number'] == 4]
dfDistrict5 = df[df['Florida District Number'] == 5]
dfDistrict6 = df[df['Florida District Number'] == 6]
dfDistrict7 = df[df['Florida District Number'] == 7]

print(df)