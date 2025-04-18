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

# read the data
insurance_test = pd.read_csv('insurance_test.csv')

# list of final column names
final_column_name = [ 
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

# dictionary to rename
rename_dict = dict(zip(list(insurance_test.columns), final_column_name))

# renaming the columns
insurance_test = insurance_test.rename(columns=rename_dict)

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
    insurance_test[i] = pd.to_datetime(insurance_test[i], format= "%d/%m/%Y")

## Creating Features
# creating Age
insurance_test['Age'] = pd.to_datetime('2019-12-31').year - insurance_test['Date_Of_Birth'].dt.year

# creating years_driving
insurance_test['Years_Driving'] = pd.to_datetime('2019-12-31').year - insurance_test['Date_Of_DL_Issuance'].dt.year

# creating car age
insurance_test['Car_Age'] = pd.to_datetime('2019-12-31').year - insurance_test['Yr_Vehicle_Registration']

# creating time since renewal
insurance_test['Time_Since_Last_Renewal'] = pd.to_datetime('2019-12-31').year - insurance_test['Date_Last_Renewal'].dt.year

# creating a non-payment termination flag
insurance_test['Non_Payment_Termination'] = 0
insurance_test.loc[insurance_test['Policies_Terminated_Non_Payment'] > 0, 'Non_Payment_Termination'] = 1

# creating non continuation of insurance flag
insurance_test['Non_Continuation_Insurance_Flag'] = 0
insurance_test.loc[insurance_test['Date_Next_Renewal'] < pd.to_datetime('2018-12-31'), 'Non_Continuation_Insurance_Flag'] = 1

# creating a new flag of new licence
insurance_test['New_License'] = (insurance_test['Years_Driving'] < 2).astype(int)

# creating vehicle age category
insurance_test['Car_Age_Cat'] = pd.cut(
    insurance_test['Car_Age'],
    bins=[-1,3,7,15,np.inf],
    labels=['New','Recent','Standard','Old']
)

# Creating premium to car value
insurance_test['Ratio_Premium_Car_Value'] = insurance_test['Premium_Amt_Current_Yr'] / insurance_test['Market_Value_EOY19']

# Creating pwoer to weight
insurance_test['Power_Wt_Ratio'] = insurance_test['Vehicle_Power_HP'] / insurance_test['Vehicle_Wt_Kg']

# customer loayalty
## weights based on some digging into domain knowledge
w_1 = 0.35
w_2 = 0.3
w_3 = 0.2
w_4 = 0.15

insurance_test['Customer_Loyalty'] = w_1*insurance_test['Years_Associates'] + w_2*insurance_test['Total_Policies_Entity']\
      + w_3*insurance_test['Max_Policy_Simultaneous_Force'] + w_4*insurance_test['Max_Product_Simultaneous_Held']

# creating young_bhp_risk
insurance_test['New_Bhp_Risk'] = insurance_test['New_License'] * insurance_test['Vehicle_Power_HP']

# year_driving_start_date
insurance_test['Years_Driving_At_Start_Date'] = (insurance_test['Start_Date_Contract'].dt.year - insurance_test['Date_Of_DL_Issuance'].dt.year)

# young driver flag
insurance_test['Young_Driver'] = 0
insurance_test.loc[insurance_test['Age'] < 25, 'Young_Driver'] = 1

# young_driver_bhp
insurance_test['Young_Bhp_Risk'] = insurance_test['Young_Driver'] * insurance_test['Vehicle_Power_HP']

# saving it in another file
insurance_test.to_csv('cleaned_test.csv')
insurance_test.to_pickle('cleaned_test.pkl')