
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#put the location of the file bank-full.csv

#---------------------------------------------------------------------
bankData = pd.read_csv(r'/bank-additional/bank-full.csv', sep = ";")
bankData['month']=bankData['month'].map({"jan": 1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12})
weekdays=[]
#--------------------------------------------------------------------
for i in range(bankData.shape[0]):
    weekdays.append(datetime.date(2014,int(bankData.month[i]),int(bankData.day[i])).weekday())

    
bankData['weekdays']=weekdays
bankData['weekdays']=bankData['weekdays'].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
bankData['weekdays'].astype('object')

#---------------------------------------------------------------------
def roundup(x):
	x=x%10
	return x

#---------------------------------------------------------------------

bankData.age = bankData.age.apply(roundup)
#---------------------------------------------------------------------
def campaign_fix(x):
	if x >= 1 and x<=10:
		return str(x)
	else:
		return 'more than 10'

	
#---------------------------------------------------------------------

bankData.campaign = bankData.campaign.apply(campaign_fix)

#---------------------------------------------------------------------
def pdays_fix(x):
	if x==-1:
		return 'no contact'
	elif x >=0 and x<=400:
		return 'within 400 days'
	else:
		return 'more than 400 days'

	
#---------------------------------------------------------------------
bankData.pdays=bankData.pdays.apply(pdays_fix)
#---------------------------------------------------------------------
y=bankData.pop('y')
X=bankData
 #---------------------------------------------------------------------
cat_X= pd.DataFrame(X).select_dtypes(include=['object'])
cat_X_df=pd.get_dummies(cat_X)
#---------------------------------------------------------------------
scaler = MinMaxScaler()
num_X= pd.DataFrame(X).select_dtypes(exclude=['object'])
num_X_cols=num_X.columns
scaled_num_X=pd.DataFrame(scaler.fit_transform(num_X))
#--------------------------------------------------------------------
X=pd.concat([cat_X_df,scaled_num_X], axis=1)
#--------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#--------------------------------------------------------------------
model = RandomForestClassifier()
model.fit(X_train,y_train)
RandomForestClassifier()
trainScore = model.score(X_train,y_train)
testScore = model.score(X_test,y_test)
print("Train Score: ", trainScore)
print("Test Score: ", testScore)


