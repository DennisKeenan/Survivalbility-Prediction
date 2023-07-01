import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import re
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as img
from subprocess import check_call
from PIL import Image,ImageDraw,ImageFont

# Survival (0=Died, 1=Survived)
# Passenger Class (1=First Class, 2=Second Class, 3=Third Class)
# Sib Sp (Number of sibling of the passenger)
# Parch (Number of parents of the passenger)
# Fare (Price of the ticket, in dollar ($))
# Embarked (C=Cherbourg,S=Southampton,Q=Queenstown)

# Read Data
train=pd.read_csv("titanic_train.csv")
test=pd.read_csv("titanic_test.csv")
train_data_backup=train.copy()
test_id=test["PassengerId"]
full_data=[train,test]
# print(train.head())
# print(test.head())
# print(test_id)

# Edit Data
    # Had a cabin or not
train["HadCabin"]=train["Cabin"].apply(lambda x:0 if type(x)==float else 1) 
test["HadCabin"]=test["Cabin"].apply(lambda x:0 if type(x)==float else 1)
    # Total number of family member
train["FamilySize"]=train["SibSp"]+train["Parch"]
test["FamilySize"]=test["SibSp"]+test["Parch"]
    # Alone passengers
train["IsAlone"]=0
train.loc[train["FamilySize"]==1,"IsAlone"]=1
test["IsAlone"]=0
test.loc[test["FamilySize"]==1,"IsAlone"]=1
    # Null/NaN Data Clearance
train["Embarked"]=train["Embarked"].fillna("S")
test["Embarked"]=test["Embarked"].fillna("S")
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
test["Fare"]=test["Fare"].fillna(test["Fare"].median())
for i in full_data:
    age_avg=i["Age"].mean()
    age_std=i["Age"].std()
    age_null_count=i["Age"].isnull().sum()
    age_random=np.random.randint(age_avg-age_std,age_avg+age_std,size=age_null_count)
    i.loc[np.isnan(i["Age"]),"Age"]=age_random
    i["Age"]=i["Age"].astype(int)


print(train["Embarked"].head(10))
print(train["Fare"].tail(10))
print(train["Age"].isnull().sum())