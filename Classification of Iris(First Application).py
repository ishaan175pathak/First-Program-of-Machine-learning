from sklearn.datasets import load_iris
import numpy as np

iris_dataset = load_iris()
print("Keys of Dataset : \n {}".format(iris_dataset.keys()))

print(iris_dataset['DESCR'][:193])

print("Target names : \n{}".format(iris_dataset['target_names']))

print("Feature names : \n{}".format(iris_dataset['feature_names']))

print("Type of Data: \n{}".format(type(iris_dataset['data'])))

print("Shape of Data: \n{}".format(iris_dataset['data'].shape))

print("First 5 columns of the data: \n{}".format(iris_dataset['data'][:5]))

print("Type of target: \n{}".format(iris_dataset['target'].shape))

print("Target: \n{}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state = 0)

print("X-train shape:{}".format(X_train.shape))
print("Y-train shape:{}".format(y_train.shape))

print("X-test shape:{}".format(X_test.shape))
print("y-test shape:{}".format(y_test.shape))

# creating data frame from data in X-train
# label the columns using the strings in iris_dataset.features_names

import pandas as pd
import matplotlib.pyplot as plt

iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset['feature_names'])

import mglearn
# creating a scatter matrix from the dataframe,color by y-train
grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize = (15,15),marker="o",hist_kwds={'bins':20},s=60,alpha=8,cmap=mglearn.cm3)
plt.show(grr.all())

# Creating an actual Machine Learning Model 

import sklearn.neighbors as skn

knn = skn.KNeighborsClassifier(n_neighbors=1)

print(knn.fit(X_train,y_train))

# making predictions

X_new = np.array([[5,2.9,1,0.2]])

print("X_new shape:{}".format(X_new.shape))

prediction = knn.predict(X_new)
print("prediction :{}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set prediction : {}".format(y_pred))

# Calculating score using mean method 
print("Test set Score :{:.2f}".format(np.mean(y_pred==y_test)))

#Calculating score using score method
print("Test set score :{:.2f}".format(knn.score(X_test,y_test)))