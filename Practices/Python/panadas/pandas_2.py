import  pandas as pd
import numpy as np

data = pd.read_csv("survey_results_public.csv")

df = pd.DataFrame(data)
# print(df.value_counts())
# print(df.loc[0]) # this will return all data for first row => 0 row
# print(df.loc[0:5 , "Hobbyist"])
# for x in df.columns[0:5]:
#     print(f"Columns Name  ==>> {x} ")
#     print(df.loc[: , x].value_counts())
#     print("================")
# === Changing indexing
# df.set_index(np.arange(1 , len(df)+1) , inplace=True)
# print(df.index)

county_thses = ['United States'  , 'Australia']
# filer_salary =( df.loc[: ,'ConvertedComp'] > 150000)
filer_salary =( df.loc[: ,'ConvertedComp'] > 150000).isin(county_thses)
col_need = ['Employment', 'Country', 'DevType' , 'YearsCode' , 'CareerSat' , 'ITperson' , 'Sexuality']

fltered_resutl = df.loc[filer_salary , col_need]
for x in fltered_resutl.columns:
    print(f"Columns Name   =>  {x} ")
    print(fltered_resutl.loc[: , x].value_counts())
    print("=================")






