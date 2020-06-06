# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:27:10 2020

@author: Sridatta reddy
"""
#import all libraries that we need in starting the project 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_csv("train.csv")

#to know with what we are dealing with
dataframe.info()
print(dataframe.columns.values)

# The columns Name, Ticket, Cabin, Embarked and Sex has object datatypes and rest 
#int and float as data types

#Find the null values if any in our DataFrame
dataframe.isnull().values.any()

train_set=pd.read_csv("train.csv" )
train_set=train_set.select_dtypes(include=['float64','int64','object'])
# train.info()

test=pd.read_csv("test.csv")
# tit2=test.select_dtypes(include=['float64','int64','object'])
# test.info()

#print(df[df["Money_Value"].isnull()][null_columns])
#tit2['Survived'] = np.nan


#Data Analysis 
#survived based on Sex

sns.countplot(train_set['Survived'], hue = train_set['Sex'])
Dead, lives = train_set.Survived.value_counts()
male, female = train_set.Sex.value_counts()
print("Percentage of Male on ship:", round(male/(male+female)*100) )
print("Percentage of Female on ship:", round(female/(male+female)*100 ))

"""65% of people boarded the ship are male and rest are Female
also Females survived more than the males. 
The death rate is way higher for males """

# survived based on Passenger Class

#find out how many classes on ship
train_set.Pclass.unique()
#so there are 3 classes on ship
train_set.Pclass.value_counts()
#in which 1st, 2nd and 3rd class has 216, 184 and 491 respectively 
#The graph shows the most people died in 3rd class which is obvious from the
#number of people who bought 3rd class tickets are high
sns.countplot(train_set['Pclass'], hue = train_set['Survived'])
t_p = train_set.groupby('Pclass')['Survived']
print(t_p.sum())
#this shows according to passenger class and the total no of ppl who survived
# on each class almost same number of ppl survived


#The Embarked class does not give much info other than S class embarkment has 
#ppl from all different classes
sns.countplot(train_set['Embarked'], hue = train_set['Survived'])


sns.distplot(train_set['Fare'], fit = 'norm')

sns.heatmap(train_set.corr(), annot = True, fmt = '.1g')
df_corr = train_set.corr().abs()
high_corr_var=np.where(df_corr>0.8)



#to fill the missing values in different columns starting with age
dataframe['Age'].fillna(round(dataframe['Age'].mean()), inplace = True)
test['Age'].fillna(round(test['Age'].mean()), inplace = True)



#train.loc[train.Age.isnull(),'Age']=train.groupby("Pclass").Age.transform('median')
#test.loc[test.Age.isnull(),'Age']=test.groupby("Pclass").Age.transform('median')


train_set = dataframe.drop( ['Name','Cabin', 'Ticket', ], axis = 1)
test_set = test.drop( ['Name','Cabin', 'Ticket', ], axis = 1)
test['Survived'] = np.nan

train_set = train_set.dropna()
#train = train.drop(columns = ['Ticket'])
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

#Selecting the dependent and independent variables
y = train.iloc[:, 1].values
X = train.iloc[:,2:  ].values

#for train
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#for test 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [7])], remainder='passthrough')
test = np.array(ct.fit_transform(test))
print(test)


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 4 ] = le.fit_transform(X[:,4])

#for test
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
test[:, 5 ] = le.fit_transform(test[:,5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20 )

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion= 'entropy', n_estimators = 500 , )
model.fit(X, y)
y_pred = model.predict(test)


from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print('acc = ', acc )

from sklearn.svm import SVC
model = SVC(kernel = 'rbf')
model.fit(X, y)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print('acc = ', acc )


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print('acc = ', acc )


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print('acc = ', acc )


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
print('acc = ', acc )
    
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
