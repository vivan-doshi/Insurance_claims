# importing important libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import linear_model
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

# changing directory
os.chdir(os.getcwd()[:-19]+'/01_Data')

# reading data
insurance_data = pd.read_csv('insurance_train.csv')

# data manipulation
# based on description
final_column_name = ['ID', 
               'Start_Date_Contract', 
               'Date_Last_Renewal', 
               'Date_Next_Renewal', 
               'Date_Of_Birth', 
               'Date_Of_DL_Issuance', 
               'Issurance_Broker_Agent_Channel', 
               'Years_Associates',
               'Total_Policies_Entity',
               'Max_Policy_Simultaneous_Force',
               'Max_Product_Simultaneous_Held',
               'Policies_Terminated_Non_Payment',
               'Half_Yearly_Payment_Method',
               'Premium_Amt_Current_Yr',
               'Total_Cost_Claims_Current_Yr',
               'Total_Number_Claims_Current_Yr',
               'Total_Number_Claims_Entire_Duration',
               'Ratio_Claims_Total_Duration_Force',
               'Motorbikes_Vans_Cars_Agricultural',
               'Rural_Urban_Flag',
               'Multiple_Drivers_Regular_Flag',
               'Yr_Vehicle_Registration',
               'Vehicle_Power_HP',
               'Cylinder_Capacity',
               'Market_Value_EOY19',
               'Vehicle_Doors',
               'Energy_Source',
               'Vehicle_Wt_Kg'
               ]

# saving old column names
old_column_names = list(insurance_data.columns)

# updating columns
insurance_data.columns = final_column_name

# making date by parsing
date_columns = [
    'Start_Date_Contract',
    'Date_Last_Renewal', 
    'Date_Next_Renewal', 
    'Date_Of_Birth', 
    'Date_Of_DL_Issuance',
]

# putting it -- in data format
for i in date_columns:
    insurance_data[i] = pd.to_datetime(insurance_data[i], format= "%d/%m/%Y")

# Creating LC and HALC variables
insurance_data['Loss_Cost'] = insurance_data['Total_Cost_Claims_Current_Yr']/insurance_data['Total_Number_Claims_Current_Yr']
insurance_data['Historically_Adjusted_Loss_Cost'] = insurance_data['Loss_Cost'] * insurance_data['Ratio_Claims_Total_Duration_Force']

# creating claim status
insurance_data['Claim_Status'] = 0
insurance_data.loc[insurance_data['Total_Number_Claims_Current_Yr']>0,'Claim_Status'] = 1

# creating Age
insurance_data['Age'] = pd.to_datetime('2019-12-31').year - insurance_data['Date_Of_Birth'].dt.year

# creating years_driving
insurance_data['Years_Driving'] = pd.to_datetime('2019-12-31').year - insurance_data['Date_Of_DL_Issuance'].dt.year

# creating car age
insurance_data['Car_Age'] = pd.to_datetime('2019-12-31').year - insurance_data['Yr_Vehicle_Registration']

# creating time since renewal
insurance_data['Time_Since_Last_Renewal'] = pd.to_datetime('2019-12-31').year - insurance_data['Date_Last_Renewal'].dt.year

# creating a non-payment termination flag
insurance_data['Non_Payment_Termination'] = 0
insurance_data.loc[insurance_data['Policies_Terminated_Non_Payment'] > 0, 'Non_Payment_Termination'] = 1

# creating non continuation of insurance flag
insurance_data['Non_Continuation_Insurance_Flag'] = 0
insurance_data.loc[insurance_data['Date_Next_Renewal'] < pd.to_datetime('2018-12-31'), 'Non_Continuation_Insurance_Flag'] = 1

# creating a new flag of new licence
insurance_data['New_License'] = (insurance_data['Years_Driving'] < 2).astype(int)

# creating vehicle age category
insurance_data['Car_Age_Cat'] = pd.cut(
    insurance_data['Car_Age'],
    bins=[-1,3,7,15,np.inf],
    labels=['New','Recent','Standard','Old']
)

# Creating premium to car value
insurance_data['Ratio_Premium_Car_Value'] = insurance_data['Premium_Amt_Current_Yr'] / insurance_data['Market_Value_EOY19']

# Creating pwoer to weight
insurance_data['Power_Wt_Ratio'] = insurance_data['Vehicle_Power_HP'] / insurance_data['Vehicle_Wt_Kg']

# customer loayalty
## weights based on some digging into domain knowledge
w_1 = 0.35
w_2 = 0.3
w_3 = 0.2
w_4 = 0.15

insurance_data['Customer_Loyalty'] = w_1*insurance_data['Years_Associates'] + w_2*insurance_data['Total_Policies_Entity']\
      + w_3*insurance_data['Max_Policy_Simultaneous_Force'] + w_4*insurance_data['Max_Product_Simultaneous_Held']

# creating young_bhp_risk
insurance_data['Young_Bhp_Risk'] = insurance_data['New_License'] * insurance_data['Vehicle_Power_HP']

# year_driving_start_date
insurance_data['Years_Driving_At_Start_Date'] = (insurance_data['Start_Date_Contract'].dt.year - insurance_data['Date_Of_DL_Issuance'].dt.year)

insurance_data.to_csv('cleaned_data.csv')
insurance_data.to_pickle('cleaned_data.pkl')