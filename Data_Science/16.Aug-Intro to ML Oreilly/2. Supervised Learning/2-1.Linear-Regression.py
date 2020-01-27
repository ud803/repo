'''
Linear models are a class of models that are widely used in practice and have been studied extensively in the last few decades. It makes a prediction using a linear function of the input features.

* 회귀 분석(Regression analysis)
 : 관찰된 연속현 변수들에 대해 두 변수 사이의 모형을 구한뒤 적합도를 측정해 내는 분석 방법

'''



'''
                    1. Linear models for Regression
'''



'''
일반적으로, 선형 모델의 예측 모형은 다음과 같다.
    y = w[0]*x[0] + w[1]*x[1] + ... + w[p]*x[p] + b
여기서, x는 feature를 나타내고, w와 b는 learning을 통해 얻어지는 계수들이다. 그리고 y hat은 모델이 만들어내는 예측 값이다.

만약 feature가 1개인 모형이라면,
    y = w[0]*x[0] + b의 꼴을 띠게 되고, 이는 고등학교 때 배운 직선의 방정식과 같다.
'''

# feature가 1개인 wave dataset에 위 모형을 적용하면, w[0]과 b의 값이 나온다.
import matplotlib.pyplot as plt
import numpy as np
import mglearn

mglearn.plots.plot_linear_regression_wave()


'''
하지만 그림을 보게 되면, 대략적인 추세만을 알 수 있고 세부적인 데이터들은 반영이 되지 않는다.
    # 우리의 목표치 y가 선형 관계에 있다는 것은 매우 강한 가정이다.
    # 그럼에도, feature가 많은 데이터셋에는 선형 모델이 매우 강력할 수 있다.

회귀 분석을 위한 선형 모델은 다양한 종류가 있지만, 그들간의 차이는 w와 b가 어떤 식으로 습득 되는지에 달려있다. 이제 가장 유명한 선형 모델들을 살펴볼 것이다.
'''


#1. Linear regression (aka ordinary least squares)
    # 선형 회귀(OLS)는 가장 단순하고 클래식한 방법이다.
    # 이 방법은 예측치와 실제 타겟 사이의 제곱의 평균을 최소화하는 w와 b를 찾는다.
    # mean squared error는 (예측치 - 실제치)^2 이다.
    # 선형 회귀에는 매개변수가 없는데, 장점이기도 하지만 모델의 복잡도를 통제할 방법이 없다.


## Wave
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train,y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
    # slope(w)는 coef_에, offset,intercept(b)는 intercept_에 저장되어 있음!
    # coef는 (1,n)의 numpy array이고, intercept는 소수이다.

print("\nWave Score\n")
print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))
    # 결과 값은 각각 0.67, 0.66
    # R 스퀘어 값이 0.66인 것은 좋지 않다, 하지만 테스트 셋과 트레이닝 셋의 점수가 비슷한 것을 알 수 있는데, 이는 우리가 underfitting 하다는 것을 의미한다!
    # 이것과 같은 일차원 데이터는 overfitting의 위험도가 낮다. (모델이 간단해서)
    # 하지만 고차원 데이터일수록 linear의 힘이 강해지고, overfitting의 위험도도 커진다.
    # 아래에서는 Boston 예제를 살펴볼 것이다.


## Boston
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train,y_train)

print("\nBoston Score\n")
print("Training set score: {:.2f}".format(lr.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))
    # 각각 0.95, 0.61로 트레이닝 셋에는 적합하지만, 테스트 셋에선 낮은 점수를 보인다.
    # 두 점수 차이는 overfitting의 명확한 신호이고, 우리는 다른 모델을 찾을 필요가 있다.
    # 그 대안은 Ridge Regression 이다.





#2. Ridge regression
    # 리지 회귀는 회귀 분석을 위한 또다른 선형 모델이기 때문에 공식 자체는 OLS와 같다.
    # 하지만, 리지 회귀에서는, 트레이닝 셋을 잘 예측하고 추가로 다른 제약 조건도 만족하도록 계수(w)들이 정해진다.
    # 또한 계수의 크기가 가능한 작기를 원한다. (w의 계수가 0이 가깝도록)
    # 다시 말하면, 각 feature는 잘 예측은 하면서도 결과에 최대한 적은 영향을 미쳐야 한다는 것을 의미한다
    # 여기서 제약이란 "regularization"이라 불리는 것인데,
    # 이는 overfitting을 피하기 위해 모델을 제한시키는 것을 의미한다.
    # 그 방법은 L2 regularization이라고 불리기도 한다.


from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train,y_train)
print("\nBoston Ridge Regression Score\n")
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
    # 0.89 0.75
    # 결과를 보면, LR보다 트레이닝 점수는 낮지만, 테스트 점수는 더 높다.
    # 이는 우리의 예상처럼, LR은 overfitting 하고 있었음을 의미한다.
    # 리지는 더 제한된 모델이기에, overfit 할 확률이 더 낮다.
    # 덜 복잡한 모델은 트레이닝 셋에서는 낮은 성과를 보이지만, 더 일반화가 잘되어 있다.




''' Alpha Parameter
리지 모델은 모델의 복잡도와 트레이닝 셋에서의 성과 사이에 트레이드 오프를 갖는다.
이 둘 사이의 가중치는 사용자에 의해 "alpha parameter"를 이용해 지정할 수 있다.
위의 예에서는 1.0이라는 기본 파라미터를 사용했다. 하지만 이게 최적의 값은 아니다.
alpha의 초적의 값은 우리가 사용하는 데이터셋에 따라 다르다.

alpha를 늘리면 계수(w)가 더 0으로 가고, 트레이닝 스코어는 낮춘다.
하지만 이는 generalization(일반화)는 돕는다. 아래의 예를 살펴보자.
'''
ridge10 = Ridge(alpha=10).fit(X_train,y_train)
print("\nAlpha10 Ridge\n")
print("Training set score: {:.2f}".format(ridge10.score(X_train,y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
    # 결과는 0.79, 0.64
    # training의 score가 감소했다.


ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
print("\nAlpha0.1 Ridge\n")
print("Training set score: {:.2f}".format(ridge01.score(X_train,y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
    # 0.93 0.77
    # 알파를 낮추면 계수가 덜 제한된다. 알파가 매우 낮아지면 선형 회귀를 닮은 모형이 된다.
    # 현재 값에서 알파를 조금씩 낮춰서 일반화를 더 향상시킬 수 있다.



plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()
    # 또한 coef을 관찰하여 alpha가 모델에 미치는 영향을 볼 수 있다.
    # 알파가 크면 더 제한된 모델이고, coef는 상대적으로 낮아진다.
    # 가로축은 인덱스, 세로축은 크기를 나타낸다.
    # alpha=10에서 Linear Regression으로 갈수록 그 계수가 커진다.



# 정규화의 영향을 보기 위한 또 다른 방법은, alpha를 고정시키고 트레이닝 셋을 변화시키는 것이다.
# 아래에서는 기존 Boston 데이터를 subset으로 쪼개어 그 크기에 따른 변화를 볼 것이다.
# ** 데이터셋 크기에 따른 모델의 성능을 나타내는 도표를 learning curve라고 한다!

mglearn.plots.plot_ridge_n_samples()
plt.show()
    # 예상 가능하듯이, 모든 사이즈에서 트레이닝 스코어가 테스트 스코어보다 높다.
    # 리지는 정규화 되었기 때문에, LR보다 트레이닝 스코어가 낮다.
    # 하지만 테스트 스코어는 더 높은데, 특히 사이즈가 작을수록 그렇다.
    # 400 미만에서 LR는 아무것도 배우지 못한다.
    # 데이터가 축적됨에 따라 두 모델의 성능은 다 증가하고, 마지막에 LR과 리지는 같아진다.
    # 즉, 트레이닝 데이터가 많으면 정규화가 덜 중요해지고, 리지와 LR의 성능차는 거의 없다!






#3. Lasso (Least Absolute Shrinkage and Selection Operator)
    # LR을 정규화시킬 때, 리지의 대안은 Lasso이다.
    # 리지와 마찬가지로, 라소도 계수를 제한하여 0까지 가도록 한다.
    # 하지만 방식이 약간 다른데, 라소는 L1 정규화를 사용한다.
    # L1 정규화를 사용했을 때, 일부 데이터는 정확히 0이 된다. 즉, 무시된다는 뜻이다.
    # 이는 자동 feature selection으로도 볼 수 있다.
    # 몇몇 특성을 0으로 하는 것은 모델을 이해하기 더 쉽게 해주고 가장 중요한 특징들만을 강조해준다.


from sklearn.linear_model import Lasso

lasso = Lasso().fit(X_train,y_train)
print("\nLasso Score\n")
print("Training set score: {:.2f}".format(lasso.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test,y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
    # 0.29 0.21 / 4개의 feature 사용
    # 결과가 매우 좋지 않은데, underfitting이라는 것을 알 수 있다.
    # 또한 105개의 feature중 오직 4개만을 사용하였다.
    # 리지와 마찬가지로, 라소도 정규화 매개변수 alpha가 있다.
    # 위는 기본 값인 1을 사용하였다. 이제 alpha를 낮춰보자.
    # max_iter 값도 올려주어야 한다.



lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train,y_train)
print("\nLasso001 Score\n")
print("Training set score: {:.2f}".format(lasso001.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test,y_test)))
print("Num of features: {}".format(np.sum(lasso001.coef_ != 0)))
    # 0.9 0.77 33
    # alpha를 낮추었더니 더 복잡한 모델에 맞춰졌다.
    # 리지보다 성능이 약간 더 좋고, 105개 중 33개의 feature만 사용했다.
    # 하지만 알파를 너무 낮게 잡으면, 정규화의 효과가 사라지고 overfitting하게 되어
    # LR과 비슷한 결과를 낸다.

lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train,y_train)
print("\nLasso00001 Score\n")
print("Training set score: {:.2f}".format(lasso00001.score(X_train,y_train)))
print("Test set score: {:.2f}".format(lasso00001.score(X_test,y_test)))
print("Num of features: {}".format(np.sum(lasso00001.coef_ != 0)))
    # 0.95 0.64 94로 그 결과가 훨씬 안좋아졌다.




# 리지와 마찬가지로 alpha에 따른 계수의 변화를 나타낼 수 있다.

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0,1.05))
plt.ylim(-25,25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
    # alpha가 1일때, 대부분의 계수가 0이고, 사라있는 계수도 크기가 작다.
    # alpha가 0.01이면, 대부분의 feature가 0이다.
    # alpha가 0.00001이 되면, 모델은 정규화되지 않는다.
    # 리지를 사용하면 모든 계수가 0이 아니다.


'''
실제 연구에서, 리지와 라소 중 리지가 우선적으로 선택된다. 하지만, feature가 많고 중요한 몇 가지만 추려내고 싶을때 라소가 더 나은 선택일 수 있다. 마찬가지로, 이해하기 쉬운 모델을 원한다면 라소가 그런 측면에서는 더 낫다.
scikit-learn은 ElasticNet이라는 클래스를 제공하는데, 이는 리지와 라소의 장점만을 결합한 것이다. 실제로는 이 결합이 가장 잘 작동하지만, L1과 L2 정규화를 위한 파라미터를 둘 다 설정해줘야 하는 번거로움이 있다.
'''
