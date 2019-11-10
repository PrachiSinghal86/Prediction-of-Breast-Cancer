
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import dataset
dataset=pd.read_csv('breast-cancer-wisconsin.data',delimiter=",",na_values='?')
dataset = dataset.fillna(0)
X = dataset.iloc[:, 1:-1].values
y=dataset.iloc[:,10].values

#data splitting

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


#Fitting the model
from sklearn.svm import SVC
classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
#predicting test set
y_pred=classifier.predict(X_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)



