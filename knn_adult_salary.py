# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

column_names = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status',
                'occupation', 'relationship', 'race',
                'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country',
                'salary']

dataset = pd.read_csv('adult_salary.csv', names = column_names, na_values = ' ?')

#to check for nan values in the dataset
dataset.isnull().sum()

X = dataset.iloc[:, 0:14]
y = dataset.iloc[:, -1]

temp_data = X.iloc[:,[1,6,13]]

#Replacing the nan values in the following columns with the mode value of that column
temp_data['workclass'] = temp_data['workclass'].fillna(temp_data['workclass'].value_counts().index[0])
temp_data['occupation'] = temp_data['occupation'].fillna(temp_data['occupation'].value_counts().index[0])
temp_data['native-country'] = temp_data['native-country'].fillna(temp_data['native-country'].value_counts().index[0])

#checking if the operation of filling nan values went successful or not
temp_data.isnull().sum()

#Putting complete data without nan values back to X
X.iloc[:,[1,6,13]] = temp_data

#Categorical cols are: [1, 3, 5, 6, 7, 8, 9, 13]

from sklearn.preprocessing import LabelEncoder
lab_enc = LabelEncoder()

#X has to be converted to object datatype from pandas DataFrame to use LabelEncdoer on it
X = X.values

for i in [1, 3, 5, 6, 7, 8, 9, 13]:
     X[:, i] = lab_enc.fit_transform(X[:, i].astype(str))   #astype(str) to process each of the values as string
    
y = lab_enc.fit_transform(y.astype(str))

#Now we need to scale the data for better optimized processing and results
from sklearn.preprocessing import StandardScaler
std_scl = StandardScaler()

X = std_scl.fit_transform(X)

#Now our data has been processed completely and we can apply ML algos on it to get prediction

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
%time knn.fit(X, y) #0.851s

%time knn.score(X, y) #87.9%, 19.8s

%time y_pred = knn.predict(X) #19.9s

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
# cm = [[23194 1526]
#       [2414  5427]]
