#!/usr/bin/env python
# coding: utf-8

# In[510]:


get_ipython().run_cell_magic('javascript', '', "\nJupyter.keyboard_manager.command_shortcuts.add_shortcut('r', {\n    help : 'run all cells',\n    help_index : 'zz',\n    handler : function (event) {\n        IPython.notebook.execute_all_cells();\n        return false;\n    }}\n);")


# In[520]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
import statsmodels.api as sm
import seaborn as sns
import loess
import random
import os
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[521]:


years=range(2000,2019)


# In[522]:


# # Ignore this section.
# # Defining Virus tramission temperarture range(in Fahrenheit)
# a=60
# b=95
# # Converting Fahrenheit to Kelvin
# a=(a-32)*5/9 + 273
# b=(b-32)*5/9 + 273


# Assigning bin classes to every instance

# In[523]:


#Import main WNV data
path_data="/Users/sparshagarwal/Downloads/WNV_challenge/neurownv_by_county.csv"
data=pd.read_csv(path_data)
data.head()


# In[524]:


#Bin labels for classification
bins={1:"[0,1)",2:"[1,6)", 3:"[6,11)", 4:"[11,16)", 5:"[16,21)", 6:"[21,26)", 7:"[26,31)", 8:"[31,36)", 9:"[36,41)", 10:"[41,46)", 11:"[46,51)", 12:"[51,101)", 13:"[101,151)", 14:"[151,201)", 15:"[201,1000)"}


# In[525]:


#Assigning bin labels to instances
bin_assign=[]
bin_values=list(bins.values())
for i in range(len(data)):
    for j in range(len(bin_values)):
        bi=bin_values[j]
        bi_last=int(bi[:-1].split(",")[1])
        if(data.iloc[i]["count"]<bi_last):
            bin_assign.append(j+1)
            break
data["bin"]=bin_assign
data.head()


# In[528]:


#Dataframe for bins and labels. Used just for visualizing bins.
bin_df=pd.DataFrame()
bin_df["Class"]=bins.keys()
bin_df["Bin"]=bins.values()
bin_df


# In[529]:


data.to_csv("/Users/sparshagarwal/Desktop/NCSA/Dataframes/WNV_challenge.csv", index=False)


# In[ ]:





# Converting the bins into one-hot vector notation

# In[530]:


#Loading the data with State Avg column

path_data="/Users/sparshagarwal/Desktop/NCSA/Dataframes/WNV_challenge_neighCountyAvg.csv"
data=pd.read_csv(path_data)

#Changes/Cleaning that needs to be done in data as mentioned by the cdc manual

data['county'] = np.where(data['county']=='Bedford/Bedford City', 'Bedford', data['county'])
data['fips'] = np.where(data['fips']=='51019/51515', '51019', data['fips'])
data['location'] = np.where(data['location']=='Virginia-Bedford/Bedford City', 'Virginia-Bedford', data['location'])

data['county'] = np.where(data['county']=='Oglala Lakota/Shannon', 'Oglala Lakota', data['county'])
data['fips'] = np.where(data['fips']=='46102/46113', '46102', data['fips'])
data['location'] = np.where(data['location']=='South Dakota-Oglala Lakota/Shannon', 'South Dakota-Oglala Lakota', data['location'])

#converting fips from object/string to int datatype

data['fips']=data['fips'].apply(pd.to_numeric)
data.rename(columns={'fips':'GEOID'}, inplace=True)
data.head()


# In[531]:


#For converting bin labels to dummy variables
bin_dummy={1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[], 15:[]}
for i in range(len(data)):
    cl=data.iloc[i]["bin"]
    for j in bin_dummy:
        bin_dummy[j].append(0)
    bin_dummy[cl][-1]=1


# In[532]:


for i in bin_dummy:
    data[i]=bin_dummy[i]


# In[533]:


data.head()


# In[ ]:





# Adding gini index data

# In[536]:


path_data_var="/Users/sparshagarwal/Downloads/NARR_weather_data/weekly_precipitation_gini.csv"
data_var=pd.read_csv(path_data_var)
data_var.head()


# In[537]:


#Adding gini index data to the main data frame
gini=[]
temp=data[data["year"]==2000]
for i in range(len(temp)):
    geoid=temp.iloc[i]["GEOID"]
    gini.extend(data_var[data_var["GEOID"]==geoid].values[0][1:])
data["Gini"]=gini
data.head()


# In[216]:


#This are the missing values in socioeconomic data
data_var[data_var.isnull()["2000"]==True]


# In[ ]:





# Adding air, precipitation and humidity data.

# In[255]:


#Defining number of days in each year
days_dict={}
for year in years:
    days=365
    if(year%4==0):
        days=366
    days_dict[year]=days


# In[538]:


pd.read_csv("/Users/sparshagarwal/Downloads/NARR_weather_data/air.sfc_complete.csv").head()


# In[539]:


pd.read_csv("/Users/sparshagarwal/Downloads/NARR_weather_data/apcp_complete.csv").head()


# In[540]:


pd.read_csv("/Users/sparshagarwal/Downloads/NARR_weather_data/rhum.2m_complete.csv").head()


# The code just below this takes a lot of time, and therefore is commented out so that it does not run. The results obtained after running the code below are stored in a csv file/dataframe which is imported everytime when it is to be used.

# In[314]:


# #Taking the average values of the variables across a single year
# df_temp=pd.DataFrame()
# df_prec=pd.DataFrame()
# df_hum=pd.DataFrame()

# variables=["Temp","Prec","Hum"]

# for var in variables:
#     value=[]
#     if(var=="Temp"):
#         path_data_var="/Users/sparshagarwal/Downloads/NARR_weather_data/air.sfc_complete.csv"
#     if(var=="Prec"):
#         path_data_var="/Users/sparshagarwal/Downloads/NARR_weather_data/apcp_complete.csv"
#     if(var=="Hum"):
#         path_data_var="/Users/sparshagarwal/Downloads/NARR_weather_data/rhum.2m_complete.csv"
#     data_var=pd.read_csv(path_data_var)
    
#     year_col=[]
#     geoid_col=[]
#     for i in range(len(data_var)):
#         print(i)
#         count=1
#         for year in years:
#             geoid_col.append(data_var.iloc[i][0])
#             year_col.append(year)
#             days=days_dict[year]
    
#             avg=np.mean(list(data_var.iloc[i][count:count+days]))
#             value.append(avg)
            
#             count+=days
    
#     if(var=="Temp"):
#         df_temp["GEOID"]=geoid_col
#         df_temp["year"]=year_col
#         df_temp["Temp"]=value
        
#     if(var=="Prec"):
#         df_prec["GEOID"]=geoid_col
#         df_prec["year"]=year_col
#         df_prec["Prec"]=value
        
#     if(var=="Hum"):
#         df_hum["GEOID"]=geoid_col
#         df_hum["year"]=year_col
#         df_hum["Hum"]=value
        


# In[542]:


df_temp.head()


# In[543]:


#Merging all the dataframes together
variables=["Temp","Prec","Hum"]
for var in variables:
    if(var=="Temp"):
        df_c=df_temp.copy()
    if(var=="Prec"):
        df_c=df_prec.copy()
    if(var=="Hum"):
        df_c=df_hum.copy()
        
    data=pd.merge(data, df_c,  how='inner', left_on=['GEOID','year'], right_on = ['GEOID','year'])


# In[546]:


data.head()


# In[ ]:





# Adding County_type column (urban/rural)

# In[547]:


# Path to the file with WNV data used to classify urban and rural counties
wnv_file_path= "/Users/sparshagarwal/Downloads/WMV_data/Arbovirus_risk_modeling_US/WNV_human_cases/WNV_NI_NNI_1999to2015_prevalence_incidence_final_20180530.csv"
#Adding County_type column
data_old=pd.read_csv(wnv_file_path, encoding='latin-1')
data_old.rename(columns={'GEOID10':'GEOID', 'Select_County':'County_type'}, inplace=True)
data_old=data_old[["GEOID","County_type"]]
data=pd.merge(data, data_old,  how='inner', left_on=['GEOID'], right_on = ['GEOID'])


# In[549]:


data.head()


# In[552]:


# #To store the data so that so taht socioeconomic data does not have be to added again.
# data.to_csv("/Users/sparshagarwal/Desktop/NCSA/Dataframes/Data_merged.csv", index=False)


# In[554]:


data=pd.read_csv("/Users/sparshagarwal/Desktop/NCSA/Dataframes/Data_merged.csv")
data.head()


# In[ ]:





# In[558]:


pd.read_csv("/Users/sparshagarwal/Downloads/WMV_data/Socioeconomics/se_data.csv").head()


# In[559]:


pd.read_csv("/Users/sparshagarwal/Downloads/WMV_data/Socioeconomics/race_data.csv").head()


# Adding socioeconomic data for 10 years. The code below is run only if socio economic data is added. Adding socioeconomic data reduces the year range from 2000-2018 to 2000-2009.

# In[556]:


variables=["Resident_population_White_alone_percent", "Median_Household_Income", "Poverty_percent_of_people"]
years=range(2000,2010)
for variable in variables:
    if(variable=="Poverty_percent_of_people" or variable=="Median_Household_Income"):
        data_path="/Users/sparshagarwal/Downloads/WMV_data/Socioeconomics/se_data.csv"
    if(variable=="Resident_population_White_alone_percent"):
        data_path="/Users/sparshagarwal/Downloads/WMV_data/Socioeconomics/race_data.csv"
    # Adding data for the variable
    s_data=pd.read_csv(data_path)
    s_data.rename(columns={'STCOU':'GEOID', 'YEAR':'year'}, inplace=True)
    s_data=s_data[s_data["GEOID"]!=0]
    s_data=s_data[s_data["year"]>=2000]
    s_data=s_data[["year", "GEOID", variable]]
    data=pd.merge(data, s_data,  how='inner', left_on=['GEOID','year'], right_on = ['GEOID','year'])


# In[560]:


data.head()


# In[ ]:





# In[ ]:


##This code is for adding weather data in CDD range but takes infinite time so ignore this part.

# path_data_temp="/Users/sparshagarwal/Downloads/NARR_weather_data/air.sfc_complete.csv"
# path_data_prec="/Users/sparshagarwal/Downloads/NARR_weather_data/apcp_complete.csv"
# path_data_hum="/Users/sparshagarwal/Downloads/NARR_weather_data/rhum.2m_complete.csv"
# data_temp=pd.read_csv(path_data_temp)
# data_prec=pd.read_csv(path_data_prec)
# data_hum=pd.read_csv(path_data_hum)

# df_temp=pd.DataFrame()
# df_prec=pd.DataFrame()
# df_hum=pd.DataFrame()

# temp=[]
# prec=[]
# hum=[]

# year_col=[]
# geoid_col=[]
# for i in range(len(data_temp)):
#     print(i)
#     count=0
#     for year in years:
#         geoid_col.append(data_temp.iloc[i][0])
#         year_col.append(year)
#         days=days_dict[year]
#         values_temp=[]
#         values_prec=[]
#         values_hum=[]
#         for day in range(days):
#             count+=1
#             if(data_temp.iloc[i][count]>=a and data_temp.iloc[i][count]<=b):
#                 values_temp.append(data_temp.iloc[i][count])
#                 values_prec.append(data_prec.iloc[i][count])
#                 values_hum.append(data_hum.iloc[i][count])
#         temp.append(np.mean(values_temp))
#         prec.append(np.mean(values_prec))
#         hum.append(np.mean(values_hum))

# df_temp["GEOID"]=geoid_col
# df_prec["GEOID"]=geoid_col
# df_hum["GEOID"]=geoid_col

# df_temp["Year"]=year_col
# df_prec["Year"]=year_col
# df_hum["Year"]=year_col

# df_temp["Temp"]=temp
# df_prec["Prec"]=prec
# df_hum["Hum"]=hum

    


# Constructing temporal dataframe for different number of prior years.

# In[561]:


#Number of previous years used for prediction
num_years=2


# In[562]:


#Features/columns that will be used for prediction.
features=["count","neighborCountyAvg", "Gini", "Temp", "Prec", "Hum","County_type","Resident_population_White_alone_percent", "Median_Household_Income", "Poverty_percent_of_people"]
#features=["count","neighborCountyAvg", "Gini", "Temp", "Prec", "Hum", "County_type"]


# In[563]:


#Minimum and maximum value of years that will form the target class
min_year=years[0]+num_years+1
max_year=years[-1]


# In[564]:


add_features=features.copy()
add_features.extend(["year","location"])
add_features.extend(str(i) for i in range(1,16))


# In[567]:


temp_df=data[add_features]


# In[568]:


#Removing the rows having NaN Gini index values
temp_df=temp_df[temp_df.isnull()["Gini"]==False]
temp_df.head()


# In[569]:


#Creating dataframe for training data
temporal_df=pd.DataFrame() #The final dataframe
for year in range(max_year,min_year-1,-1):
    col="A"
    yearly_df=pd.DataFrame()
    target=temp_df[temp_df["year"]==year]
    classes=[str(i) for i in range(1,16)]
    target=target[classes]
    for prior_year in range(year-2,year-num_years-2,-1):
        df_prior_year=temp_df[temp_df["year"]==prior_year]
        for feature in features:
            feat_values=list(df_prior_year[feature])
            yearly_df[col]=feat_values
            col=chr(ord(col)+1)
    yearly_df=pd.concat([yearly_df.reset_index(),target.reset_index()], axis=1).drop(["index"], axis=1)
    temporal_df=temporal_df.append(yearly_df)


# In[570]:


#Creating dataframe for testing data. Ignore this part for now.
year=2020
col="A"
temporal_df_test=pd.DataFrame()
for prior_year in range(year-2,year-num_years-2,-1):
    df_prior_year=temp_df[temp_df["year"]==prior_year]
    for feature in features:
        feat_values=list(df_prior_year[feature])
        temporal_df_test[col]=feat_values
        col=chr(ord(col)+1)


# In[572]:


#Exporting data so that it can be imported in another jupyter notebook.
temporal_df.to_csv("/Users/sparshagarwal/Desktop/NCSA/Dataframes/Temporal_df.csv", index=False)
temporal_df_test.to_csv("/Users/sparshagarwal/Desktop/NCSA/Dataframes/Temporal_df_test.csv", index=False)


# In[573]:


temporal_df.head()


# In[ ]:




