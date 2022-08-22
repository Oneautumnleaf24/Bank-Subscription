import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go


#Preprocessing
#in df put the location of the file bank-full.csv
df = pd.read_csv(r'/bank-full.csv', delimiter = ';')
print(df.info())
df.describe()
print((df.head()))
print("===========================================================")

#storing each attribute column in an attribute
age = df["age"]
job = df["job"]
marital = df["marital"]
education = df["education"]
default = df["default"]
housing = df["housing"]
loan = df["loan"]
contact = df["contact"]
day = df["day"]
month = df["month"]
duration = df["duration"]
campaign = df["campaign"]
pdays = df["pdays"]
previous = df["previous"]
poutcome = df["poutcome"]
class_variable = df["y"]

print("===========================================================")

# counting cases of Positve/Negative class cases
print((df["y"].value_counts()))
print("===========================================================")

#Formatting Binary Values
print("Converting categorical binary values into numerical binary values: (Yes:1, No:0)")
df["default"] = df["default"].map({"yes": 1, "no": 0})
df["housing"] = df["housing"].map({"yes": 1, "no": 0})
df["loan"] = df["loan"].map({"yes": 1, "no": 0})
df["y"] = df["y"].map({"yes": 1, "no": 0})

print("Validation of Converted categorical data to binary")
print((df["default"].value_counts()))
print((df["housing"].value_counts()))
print((df["loan"].value_counts()))
print((df["y"].value_counts()))
print("===========================================================")

#view of dataset
print(df)

print("===========================================================")

#correlation
corrdata = df.corr()
print(corrdata)

print("===========================================================")
f, ax = plt.subplots()
palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax = sns.boxplot(x="y", y="age", hue="y",data=df, palette= palette)
plt.legend(handles, labels,title='Subscribed' )
plt.title("Age Vs Subscribers")

print("===========================================================")
f, ax = plt.subplots()
palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax = sns.scatterplot(x="age", y="y", hue="y",data=df, palette= palette)

plt.legend(handles, labels,title='Subscribed' )
plt.title("Age Vs Subscribers")

print("===========================================================")
f, ax = plt.subplots()
palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax = sns.boxplot(x="y", y="previous", hue="y",data=df, palette= palette)
plt.legend(handles, labels,title='Subscribed' )
plt.title("previous Vs Subscribers")

print("===========================================================")
f, ax = plt.subplots()
palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

ax = sns.scatterplot(x="previous", y="y", hue="y",data=df, palette= palette)
plt.legend(handles, labels,title='Subscribed' )
plt.title("previous Vs Subscribers")

plt.show()
