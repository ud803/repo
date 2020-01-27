'''
Let's assume that a hobby botanist is interested in distinguishing the species of some iris flowers that she has found.
She has collected some measurements associated with each iris:
  the length and width of the petals
  the length and width of the sepals
She also has measurements of some irises that have been identified:
  setosa
  versicolor
  virginica

Our goal is to build a machine learning model that can learn from the measurements of these irises whose species is known, so that we can predict the species for a new iris.

Because we have measurements for which we know the correct species of Iris, this is a supervised learning problem. This is an example of a classification problem. The possible outcomes are called classes. The desired output for a single data point (an iris) is the species of the flower. For a particular data point, the species it belongs to is called its label.
'''
# 필요한 라이브러리 호출
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# iris_dataset 변수에는 Bunch object가 저장된다. 딕셔너리와 비슷하게 key와 value로 구성되어 있음

from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

print("\n\n\n\n")

print(iris_dataset['DESCR'][:193] + "\n...")

print("\n\n\n\n")
# Target name은 str, Feature name도 str
# 데이터 구성을 보기 위해 다 읽어들임. 책과 조금 다름
print("Target names: \n{}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("data: \n{}".format(iris_dataset['data']))
print("target:\n{}".format(iris_dataset['target']))

# type명령어를 통해 dataset의 data 의 타입 출력
print("Type of data: {}".format(type(iris_dataset['data'])))

# shape은 number of samples * number of features
print("Shape of data: {}".format(iris_dataset['data'].shape))

# feature values for the first five samples
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

# target array
# each species is encoded as integers from 0 to 2
# 0 means iris_dataset['target_names'][0] (setosa)
print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("\n\n\n\n")




# 2. Split Test Data


# call train_set_spit and assign the outputs
# before split, the functon shuffles the dataset using a pseudorandom number generator.
# if we just use the last 25 of the data, all the data points would have the label 2, as it is sorted that way. (so we shuffle)
# to make sure that we'll get the same output if we run the same function several times, we provide the pseudorandom number generator with a fixed seed using the random_state parameter. This will make the outcome deterministic.
# X_train 75% of the rows, X_test remaining 25%.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("\n\n\n\n")




# 3. Visualization

# WE first convert NumPy array into a pandas DataFrame.
# create dataframe fom data in X_train
# label the columns using the strings in iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=0.8, cmap=mglearn.cm3)
print("\n\n\n\n")
#plt.show()
# 위 #을 지워야 그림이 나타남!!!






# 4. Build a model


# Here we will us a k-nearest neighbors classifier, which is easy to understand.
# Building this model only consists of storing the training set. To make a prediction for a new data point, the algorithm finds the point in the training set that is closest to the new point. Then it assigns the label of this training point to the new data point.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# show string representation of classifier
knn.fit(X_train, y_train)

# put new data
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

# make a prediction
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
print("\n\n\n\n")






# 5. Evaluating the model


# 앞에서 만들어놓은 테스트 데이터로 모델을 검증

y_pred = knn.predict(X_test)
print("Test set predictions\n {}".format(y_pred))

# prediction과 정답의 평균을 구해 점수를 냄
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

# 점수 내는 2번째 방법, knn자체 함수
print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))

# 여기서는 0.97, 즉 97%의 정확도를 보임
