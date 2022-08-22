
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc

#in df put the location of the file bank-full.csv

bankData = pd.read_csv(r'/bank-full.csv',sep=";")
#---------------------------------------------------------------------------
bankData["default"] = bankData["default"].map({"yes": 1, "no": 0})
bankData["housing"] = bankData["housing"].map({"yes": 1, "no": 0})
bankData["loan"] = bankData["loan"].map({"yes": 1, "no": 0})
bankData["y"] = bankData["y"].map({"yes": 1, "no": 0})
#---------------------------------------------------------------------------
bankData.drop("duration",axis=1,inplace=True)
bankData['month']=bankData['month'].map({"jan": 1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12})
weekdays=[]
for i in range(bankData.shape[0]):
    weekdays.append(datetime.date(2014,int(bankData.month[i]),int(bankData.day[i])).weekday())

    
bankData['weekdays']=weekdays
bankData['weekdays']=bankData['weekdays'].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
bankData['weekdays'].astype('object')

#---------------------------------------------------------------------------
y=bankData.pop('y')
X=bankData
#---------------------------------------------------------------------------
cat_X= pd.DataFrame(X).select_dtypes(include=['object'])
num_X_df= pd.DataFrame(X).select_dtypes(exclude=['object'])
cat_X_df=pd.get_dummies(cat_X)
#---------------------------------------------------------------------------
X=pd.concat([cat_X_df,num_X_df], axis=1)
#---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.head()

#Random Forest

model = RandomForestClassifier()
model.fit(X_train,y_train)
RandomForestClassifier()
y_pred_rf = model.predict_proba(X_test)[:, 1]

trainScore = model.score(X_train,y_train)
testScore = model.score(X_test,y_test)

print("Train Score: ", trainScore)
print("Test Score: ", testScore)

y_pred_rf = model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
roc_auc_rf= auc(fpr_rf, tpr_rf)


#adaboost


model_ada = AdaBoostClassifier(n_estimators = 200)
model_ada.fit(X_train, y_train)
y_pred_ada = model_ada.predict_proba(X_test)[:, 1]

fpr_ada, tpr_ada, _ = roc_curve(y_test, y_pred_ada)
roc_auc_ada = auc(fpr_ada, tpr_ada)

trainScore1 = model_ada.score(X_train,y_train)
testScore1 = model_ada.score(X_test,y_test)

print("Train Score ADA: ", trainScore1)
print("Test Score ADA: ", testScore1)

#decision tree

model_dt = DecisionTreeClassifier(max_depth = 8, criterion ="entropy")
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict_proba(X_test)[:, 1]

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

trainScore2 = model_dt.score(X_train,y_train)
testScore2 = model_dt.score(X_test,y_test)
print("Train Score DT: ", trainScore1)
print("Test Score DT: ", testScore1)


plt.figure(1)
lw = 2
plt.plot(fpr_ada, tpr_ada, color='red',
         lw=lw, label='Ada Boost(AUC = %0.2f)' % roc_auc_ada)
plt.plot(fpr_rf, tpr_rf, color='darkorange',
         lw=lw, label='Random Forest(AUC = %0.2f)' % roc_auc_rf)
plt.plot(fpr_dt, tpr_dt, color='green',
         lw=lw, label='Decision Tree(AUC = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.legend(loc="lower right")


plt.show()
