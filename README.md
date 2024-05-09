# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.
   

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Preethika N
RegisterNumber:  212223040130
*/

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrices
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

Data Result

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/0665f387-00ab-434f-94c0-8e8c107f3857)

Data Head

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/97d8db41-ed82-493c-8cec-49d5079b3bb2)

Data Information

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/86a2da0e-200c-425f-ac20-8d56a8e32aca)

Null Data

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/ab7620ea-944e-4a65-8f39-7c5744c94b15)

Y Pred

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/5f2ba2b2-2535-4a8b-b6be-efbe01f49f2c)

Accuracy

![image](https://github.com/preethi2831/Implementation-of-SVM-For-Spam-Mail-Detection/assets/155142246/e4cfeab2-d0df-4fae-a29e-797be50f030f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
