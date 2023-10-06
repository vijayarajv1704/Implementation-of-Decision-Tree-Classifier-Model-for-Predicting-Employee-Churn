# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Libraries.

2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeClassifier , find accuracy and prediction for the dataset.

5.End the program.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vijayaraj V
RegisterNumber:  212222230174
*/
```
```

import pandas as pd
data=pd.read_csv("/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```


## Output:

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/5d816ffa-7cc1-491a-88cb-002c7c8ef758)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/a746cbb2-abfb-40a1-a0d2-627f8abe87c2)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/9ef26ef4-9228-4d3a-a40d-17140f1e80a6)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/2f8f9a39-71ff-4ea6-babb-20cded30536f)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/e5b72d4f-b354-4efe-9fda-65edb433f0d6)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/22a930f7-0ad8-44a3-ba23-b7d5cfb601c1)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/3cb95928-5014-4c03-809b-df012d2270ad)

![image](https://github.com/vijayarajv1704/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121303741/be6243cb-fe20-45e1-9584-129d8f1f6c05)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
