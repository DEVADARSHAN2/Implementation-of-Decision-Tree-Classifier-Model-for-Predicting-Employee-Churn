# EX-06 Implementation of Decision Tree Classifier Model for Predicting Employee Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score. 

## Program:
```
Developed by: DEVADARSHAN A S
RegisterNumber: 212222110007
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
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
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
![Screenshot 2024-04-04 134517](https://github.com/DEVADARSHAN2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119432150/1f4d10f2-eaec-4455-a3e1-9cf2a9de3ab2)
### Accuracy value
![Screenshot 2024-04-04 134040](https://github.com/DEVADARSHAN2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119432150/99c8e2f3-9146-4e94-b9dc-06205508a6cb)
### Predicted value
![Screenshot 2024-04-04 134050](https://github.com/DEVADARSHAN2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119432150/a5bc2018-b314-4c01-9110-6bde36c92320)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
