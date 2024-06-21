#___Importing all the Libraries needed for the project___#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics

#___Importing the Raw data___#

data = pd.read_csv("F:\My_python_programs\Diabetes Dataset\diabetes_prediction_dataset.csv")
#print(data)

#___Preprocessing of the inserted raw data___# 

#print(data.shape)
#iszero = data.isnull().values.any()
#print(iszero)
Parameter = data.iloc[:,2:7]
#print(Parameter)
resultant = data.iloc[:,data.columns == 'diabetes']
#print(resultant)
initial_dataset = pd.concat([Parameter,resultant],axis=1)
#print(initial_dataset)
str2int = preprocessing.LabelEncoder()
initial_dataset['smoking_history'] = str2int.fit_transform(initial_dataset['smoking_history'])
#print(initial_dataset)


#___visualising the data using matplotlib library___#

outcome_count = pd.value_counts(initial_dataset['diabetes'], sort = True).sort_index()
outcome_count.plot(kind = 'bar', color='yellow')
plt.title('diabetes histogram', fontweight='bold', fontsize='15', color='red')
plt.xlabel('diabetes', fontweight='bold', fontsize='15', color='gray')
plt.ylabel('Frequency', fontweight='bold', fontsize='15', color='gray')
#print(plt.show())

#___all columns have to standard with respect to other___# 

stan = preprocessing.StandardScaler()
initial_dataset['new_bmi'] = stan.fit_transform(initial_dataset['bmi']. values.reshape(-1,1))
final_dataset = initial_dataset.drop(['bmi'], axis = 1)
#print(final_dataset)

#___Split data for training and testing___#

final_parameter = final_dataset.iloc[:,final_dataset.columns != 'diabetes']
final_resultant = final_dataset.iloc[:,final_dataset.columns == 'diabetes']
#print(final_parameter)
#print(final_resultant)
x_train, x_test, y_train, y_test = train_test_split(final_parameter, final_resultant, test_size = 0.2)

#___Applying Logistic Regression___#

model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())
result = model.predict(x_test)
#print(result)
accuracy = model.score(x_test,y_test)
print("Accuracy using Logistic regression:", accuracy)

#___Applying Decission Tree Classifier___#

clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(x_train,y_train)
yPred = clf.predict(x_test)
#print(yPred)
scores = cross_val_score(clf, y_test, yPred, cv=10)
print("Accuracy using Decision tree:",metrics.accuracy_score(y_test, yPred))
#print(scores)

#__Applying Linear Regression__#
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
