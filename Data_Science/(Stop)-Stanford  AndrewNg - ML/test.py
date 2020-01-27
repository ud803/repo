import numpy as np
import pandas as pd


def computeCost(theta, x, y) :
    matrixX = np.array(x).reshape(len(x), -1)
    matrixY = np.array(y).reshape(len(y), -1)
    theta = np.array(theta).reshape(len(theta), -1)
    m = len(x)
    return (1/(2*m)*np.dot( (np.dot(matrixX, theta)- matrixY).T, (np.dot(matrixX, theta)- matrixY) )).item(0)


def batchGradientDescent(theta, x, y) :
    alpha = 0.001
    count = 1
    matrixX = np.array(x).reshape(len(x), -1)
    matrixY = np.array(y).reshape(len(y), -1)
    theta = np.array(theta).reshape(len(theta), -1)
    print(matrixX[:,1])

    theta = np.array(theta).reshape(len(theta), -1)
    m = len(x)
    print((np.dot(matrixX, theta) - matrixY).shape,"qwqwqw")
    print( (np.dot(matrixX, theta) - matrixY).shape)
    print(np.dot( (np.dot(matrixX, theta) - matrixY).flatten(), matrixX[:,1]))


    print(len(y))
    while(count < 20) :
        #print(computeCost(theta, x, y))

        theta[0] = theta[0] - alpha * np.sum( np.dot(matrixX, theta) - matrixY)/len(y)
        theta[1] = theta[1] - alpha * np.dot( (np.dot(matrixX, theta) - matrixY).T, matrixX[:,1])/len(y)


        print(theta)
        count += 1

from sklearn.linear_model import LinearRegression
Lr = LinearRegression().fit( batchExample[['Intercept','Living Area']], batchExample['Price'])
print(Lr.coef_)


batchExample = pd.read_csv("testcsv.csv")
batchGradientDescent([0,0.3222], batchExample[['Intercept','Living Area']], batchExample['Price'])
