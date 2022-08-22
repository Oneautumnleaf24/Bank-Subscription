import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt

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
#Bar Graph For Marital
f, ax = plt.subplots()

palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

sns.barplot(x="marital", y="y", hue="y", data=df, palette=palette, estimator=lambda x: len(x))
ax.set(ylabel="Subscription")
ax.set_xticklabels(df["marital"].unique(), rotation=0, rotation_mode="anchor")
plt.legend(handles, labels,title='Subscribed' )
plt.title("Marital Vs Subscribers")

print("===========================================================")
#Bar Graph For Job
f, ax = plt.subplots()

palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

sns.barplot(x="job", y="y", hue="y", data=df, palette=palette, estimator=lambda x: len(x))
ax.set(ylabel="Subscription")
ax.set_xticklabels(df["job"].unique(), rotation=0, rotation_mode="anchor")
plt.legend(handles, labels,title='Subscribed' )
plt.title("Job Vs Subscribers")

print("===========================================================")
#Bar Graph For Education
f, ax = plt.subplots()

palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

sns.barplot(x="education", y="y", hue="y", data=df, palette=palette, estimator=lambda x: len(x))
ax.set(ylabel="Subscription")
ax.set_xticklabels(df["education"].unique(), rotation=0, rotation_mode="anchor")
plt.legend(handles, labels,title='Subscribed' )
plt.title("Education Vs Subscribers")


print("===========================================================")
#Bar Graph For y
f, ax = plt.subplots()

palette = ["#FA5858","#64FE2E"]
colors = {'Yes':'green', 'No':'red'}
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]

sns.barplot(x="y", y="y", hue="y", data=df, palette=palette, estimator=lambda x: len(x))
ax.set(ylabel="y")
ax.set_xticklabels(df["y"].unique(), rotation=0, rotation_mode="anchor")
plt.legend(handles, labels,title='Subscribed' )
plt.title("y Vs Subscribers")


print("===========================================================")

print(df["y"].value_counts("1"))
print(df["job"].value_counts("1"))



plt.show()
