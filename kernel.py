# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:43:13 2018

@author: Antonin Villemin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------- Importing the data ------------------------------------------

dataset=pd.read_csv('train.csv')
x_test=pd.read_csv('test.csv')
all_data=pd.concat((dataset.iloc[:, 0:-1],x_test)).reset_index()
#Look at the missing data
all_data_missing = ((all_data.isnull().sum() / len(all_data)) * 100).sort_values(ascending=False)

# ---------------------------- Filling the missing data ----------------------------------------

all_data['PoolQC']=all_data['PoolQC'].fillna("NA")
all_data['MiscFeature']=all_data['MiscFeature'].fillna("NA")
all_data['Alley']=all_data['Alley'].fillna("NA")
all_data['Fence']=all_data['Fence'].fillna("NA")
all_data['FireplaceQu']=all_data['FireplaceQu'].fillna("NA")
all_data["LotFrontage"] = all_data.groupby("LotConfig")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
#There is two specific cases that we should take into account
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('NA')    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)    
#test=all_data[all_data['GarageType'].notnull() & all_data['GarageFinish'].isnull()]
#test2=all_data[all_data['BsmtCond'].isnull()]
#test3=all_data[all_data['KitchenQual'].isnull()]
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('NA')    
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("NA")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data ["MSZoning"] = all_data["MSZoning"].fillna("RL")
all_data ["Utilities"] = all_data["Utilities"].fillna("AllPub")
all_data ["Functional"] = all_data["Functional"].fillna("Typ")
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)    
all_data ["SaleType"] = all_data["SaleType"].fillna("WD")
all_data ["Electrical"] = all_data["Electrical"].fillna("SBrkr")
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])

# --------------------------------- Label Encoding -----------------------------------------------

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
#cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
#        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
#        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
#        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 
#        'YrSold', 'MoSold')

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
         'ExterQual','ExterCond','HeatingQC','KitchenQual','PoolQC',
         'BsmtFinType1','BsmtFinType2','Functional','Fence','BsmtExposure','GarageFinish','LandSlope','LotShape','PavedDrive','MSSubClass')
values = [["NA","Po","Fa","TA","Gd","Ex"]]*5+[["Po","Fa","TA","Gd","Ex"]]*4+[["NA","Fa","TA","Gd","Ex"]]+[["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"]]*2+[["Sal","Sev","Maj2","Maj1","Mod","Min2","Min1","Typ"]]
values = values+ [["NA","MnWw","GdWo","MnPrv","GdPrv"]]+[["NA","No","Mn","Av","Gd"]]+[["NA","Unf","RFn","Fin"]]+[["Sev","Mod","Gtl"]]+[["IR3","IR2","IR1","Reg"]]+[["N","P","Y"]]+[["190","180","160","150","120","90","85","80","75","70","60","50","45","40","30","20"]]

cols2 = ('Street','CentralAir','YrSold','MoSold')

for i in range(len(cols)):
    labelencoder = LabelEncoder()
    labelencoder.fit(values[i])
    all_data[cols[i]]=labelencoder.transform(all_data[cols[i]])

for col in cols2:
    labelencoder = LabelEncoder()
    all_data[col]=labelencoder.fit_transform(all_data[col].values)

#Getting dummy categorical features
all_data = pd.get_dummies(all_data)