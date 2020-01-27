import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors


# Generate Sample Data
np.random.seed(0) # generator에 seed 제공

X = np.sort(5*np.random.rand(40,1),axis=0) # np.random.rand(a,b) a,b행렬의 형태로 [0,1) 사이의 숫자 제공
T = np.linspace(0,5,500)[:, np.newaxis] #np.newaxis는 하나의 열로 만듦
    # linspace : return evenly spaced numbers over a specified interval
    # linspace(start,stop,num=50, endpoint=True, retstep=False)
y = np.sin(X).ravel()


# Add noise to targets
y[::5] += 1* (0.5 - np.random.rand(8))
    # slicing은 i:j:k의 형태로 이루어짐
    # i는 start, j는 end, k는 step


# Fit Regression Model
n_neighbors = 5

for i, weights in enumerate(['uniform','distance']):    #enumerate은 차례로 번호매김
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X,y).predict(T)

    plt.subplot(2,1,i+1)
    plt.scatter(X,y,c='k', label='data')
    plt.plot(T,y_,c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights= '%s')" %(n_neighbors, weights))

plt.show()
