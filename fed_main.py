import pandas as pd
import openpyxl
from Net import Net

input_dim = 8
hidden_dim = 64
output_dim = 1

model = Net(input_dim, hidden_dim, output_dim)

df = pd.read_excel('SchoolGrades22 PreProcessed.xlsx', sheet_name="By District")
print(df)

#dropping all empty values
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

yesNoMapping = {'NO': 1, 'YES': 2}
df['Charter School'] = df['Charter School'].map(yesNoMapping)
df['Title I'] = df['Title I'].map(yesNoMapping)

#Get rid of unneeded columns
df = df.drop('County Number', axis=1)
df = df.drop('County Name', axis=1)
df = df.drop('School Name', axis=1)

# Print the modified DataFrame
print(df)

df.to_excel("SchoolGrades22 PostProcessed.xlsx", index=False)
print(pd.read_excel("SchoolGrades22 PostProcessed.xlsx", sheet_name="Sheet1"))