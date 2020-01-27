'''
수업에 사용할 예제들 모음
'''


'''

1. Classification Data Set - forge dataset

    # An exmaple of a synthetic two-class classification dataset

    # "Forge Dataset" - two features


    # Creates a scatter plot visualizing all of the data points in the set.
    # First feature on the x-axis, second feature on the y.
    # Color and shape of the dot indicates its classes.

'''

import mglearn
import matplotlib.pyplot as plt

# generate datset
X, y = mglearn.datasets.make_forge()

#plot dataset
mglearn.discrete_scatter(X[:,0],X[:,1],y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))    #shape = 26 datapoints, 2 fts
plt.show()


'''

2. Regression Data set - wave dataset
    # single input feature, continuous taret variable
    # single feature on the x, target on the y

'''

import mglearn
import matplotlib.pyplot as plt

# generate dataset
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X,y,'o')
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
print("X.shape: {}".format(X.shape))
plt.show()




'''
3. Wisconsin Breast Cancer dataset

  # large classification example

  # 569 data points, 30 features each !

'''

import mglearn
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_breast_cancer


# Load Datasets
cancer = load_breast_cancer()

# Show Dataset Keys
print("In[4]\n")
print("cancer.keys(): \n{}".format(cancer.keys()))

# Show All Datasets related to each Key
# In[7]도 여기에 포함됨
for _ in cancer :
    print("cancer.{}: \n{}".format(_,cancer[_]))
    print("\n\n\n")

# Shape
print("In[5]")
print("Shape of cancer data: {}".format(cancer.data.shape))

# Sample counts per class
print("\n\nIn[6]")
print("Sample counts per class:\n{}".format({n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
# 위 함수에 대한 설명은 python 문서로





'''
    4. Boston Housing dataset

    # to predict the median value of homes in Boston neighborhoods

    # 506 data points, 13 features

    # Large Regression Dataset

'''

from sklearn.datasets import load_boston
import mglearn

boston = load_boston()

print("In[8]\n")
print("Data shape: {}".format(boston.data.shape))


print(boston.keys())
'''
print("target\n",boston.target)
print("data\n",boston.data)
print("names\n",boston.feature_names)
print(boston.DESCR)
'''

# 이 예제에서는 각 feature 뿐만 아니라, feature 들간의 상관관계도 보게 된다.
# we'll consider the product of two features
# Including derived feature like these is called "feature engineering"

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

# 설명에서는 13C2 의 결과 값 + 원래 13개의 값 = 104라는데 나는 91이 나온다..?
