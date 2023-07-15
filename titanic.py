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
    
    # Remove Title from Name
def get_title(Name):
    search=re.search(' ([A-Za-z]+)\.',Name)
    if search:
        return search.group(1)
    return("")
    
    # Title Revision
for i in full_data:
    i["Title"]=i["Name"].apply(get_title)
for i in full_data:
    i["Title"]=i["Title"].replace(['Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona','Lady'],'Other')
    i["Title"]=i["Title"].replace(['Mlle','Ms'],'Miss')
    i["Title"]=i["Title"].replace('Mme','Mrs')
    i["Title"]=i["Title"].replace('Sir','Mr')
# a,count=np.unique(train["Title"],return_counts=True)    
# print(a,count)

    # Mapping
unused_column=["PassengerId","Name","Ticket","Cabin","SibSp"]
for i in full_data:
    i["Sex"]=i["Sex"].map({"female":0,"male":1}).astype(int)
    i["Title"]=i["Title"].map({"Mr":1,"Master":2,"Mrs":3,"Miss":4,"Other":5})
    i["Title"]=i["Title"].fillna(0)
    i["Title"]=i["Title"].astype(int)
    i["Embarked"]=i["Embarked"].map({"C":1,"S":2,"Q":3}).astype(int)
    i.loc[i["Fare"]<7.91,"Fare"]=0
    i.loc[(i["Fare"]>7.91) & (i["Fare"]<14.454),"Fare"]=1
    i.loc[(i["Fare"]>14.454) & (i["Fare"]<31),"Fare"]=2
    i.loc[i["Fare"]>31,"Fare"]=4
    i["Fare"]=i["Fare"].astype(int)
    i.loc[i["Age"]<16,"Age"]=0
    i.loc[(i["Age"]>16) & (i["Age"]<32),"Age"]=1
    i.loc[(i["Age"]>32) & (i["Age"]<48),"Age"]=2
    i.loc[(i["Age"]>48) & (i["Age"]<64),"Age"]=3
    i.loc[i["Age"]>64,"Age"]=4
    i["Age"]=i["Age"].astype(int)
train=train.drop(unused_column,axis=1)
test=test.drop(unused_column,axis=1)
    # print(i)

    # Gini Impurity
def get_GI(survive,total):
    survival_rate=survive/total
    GI=2*(survival_rate*(1-survival_rate))
    return GI
# print(get_GI(342,549))

    # Graphs
color_map=mp.cm.viridis
mp.figure(figsize=(12,12))
mp.title("Correlation of features",y=1.05,size=15)
sb.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=color_map,linecolor="White",annot=True)
# mp.show()

    # Correlation
print(train[["Title","Survived"]].groupby(["Title"],as_index=False).agg(["mean","count","sum"]))
print(train[["Sex","Survived"]].groupby(["Sex"],as_index=False).agg(["mean","count","sum"]))
print(train[["HadCabin","Survived"]].groupby(["HadCabin"],as_index=False).agg(["mean","count","sum"]))
print(train[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).agg(["mean","count","sum"]))

    # Cross Validation
CV=KFold(n_splits=10)
accuracy=list()
