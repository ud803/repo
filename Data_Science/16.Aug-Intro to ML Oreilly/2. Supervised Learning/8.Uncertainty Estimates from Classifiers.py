'''
                Uncertainty Estimates from Classifiers
'''

'''
scikit-learn의 또 다른 장점은 classifier가 예측의 불확실한 정도를 제공한다는 것이다.
분류를 하는 사용자는 그 예측값뿐만 아니라 이게 얼마나 정확한지도 궁금할 것이다.
scikit-learn에는 "decision_function"과 "predict_proba"라는 함수를 통해 이 기능을 제공한다.
모든 classifier가 둘 다 갖고 있는 것은 아니지만, 대부분이 가지고 있다.
여기서는 그래디언트부스팅을 예시로 살펴보자.
'''

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
import numpy as np


X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

#rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y]

# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)

# build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)


print("\nX\n")
print(X.shape)
print(X_train.shape, X_test.shape)
print("\ny\n")
print(y)


#1. The Decision function
    # 이항 분류에서, decision_function의 리턴값은 shape의 형태이다.
    #그리고 각 샘플마다 하나의 부동 소수점형식의 숫자를 반환한다.

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape : {}".format(gbrt.decision_function(X_test).shape))

#show the first few entries of decision_function
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:]))
    #양의 값들은 positive class(이 경우 1)에 대한 선호도, 음의 값들은 negative (이 경우 0)에 대한 선호도를 나타낸다.
    #결정 함수의 부호만을 보고 모델의 예측값을 다시 살펴보자.
    #하지만 이는 True/False만을 보여주기 때문에 classes_를 사용해보자.

#make the boolean True/False into 0 and 1
greater_zero = (gbrt.decision_function(X_test) >0).astype(int)
# use 0 and 1 as indices into classes
pred = gbrt.classes_[greater_zero]
# pred is the same as the output of gbrt.predict
print("pred is equal to predictions: {}".format(np.all(pred == gbrt.predict(X_test))))
    #결정 함수의 범위는 임의적이고, 데이터와 모델 파라미터에 따라 다르다.


decision_function = gbrt.decision_function(X_test)
print("Decision function minimum: {:.2f} maximum: {:.2f}".format(np.min(decision_function), np.max(decision_function)))
    #모든 지점에 대한 결정 함수를 그림으로 나타내어보자.


fig, axes = plt.subplots(1, 2, figsize=(13,5))
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)

for ax in axes :
    mglearn.discrete_scatter(X_test[:,0], X_test[:,1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))
plt.show()
    #첫 번째 그림은 모든 지점에 대한 decision boundary이고
    #두 번째 그림은 모든 지점에 대한 decision function이다.
    #이러한 encoding은 얼마나 그 예측에 대해 확실한가를 그림으로 보여준다.
    #하지만 이 그림에서 둘 사이의 경계를 구분하기는 힘들다.



#2.Predicting Probabilities
    #predict_proba는 각 클래스마다 확률을 계산해주고, 결정함수보다 더 이해하기 쉽다.
    #결정함수의 모양은 (n_samples,)였지만 predict_proba는 (n_samples, 2)이다.

print("\n\n\nShape of Probabilities: {}".format(gbrt.predict_proba(X_test).shape))

#show the first few entries of predict_proba
print("\nPredicted probabilities:\n{}".format(gbrt.predict_proba(X_test[:6])))
    #각 행의 첫 번째 열은 첫 번째 클래스에 대한 확률, 두 번째도 마찬가지이다.
    #확률이기 때문에 0에서 1 사이의 값을 갖고, 둘의 합은 항상 1이다.
    #50%이상의 확률을 가진 클래스의 라벨을 부여받게 된다.

    #이 예시에서 확률이 상당히 높은 것을 볼 수 있는데,
    #그 확률의 신뢰성은 모델과 파라미터에 달려 있다.
    #더 overfit한 모델은 더 certain한 예측을 내린다. (그 예측이 틀리더라도)
    #덜 복잡한 모델은 예측이 더 uncertain하다.
    # "calibrated"
    #   만약 uncertainty가 그 예측의 정확도와 같다면, calibrate라고 부른다.
    #   즉, 70%의 가능성으로 예측한 것은 항상 70%만큼 옳을 때!

fig, axes = plt.subplots(1, 2, figsize=(13,5))

mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4, fill=True, cm=mglearn.cm2)
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1], alpha=.5, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    mglearn.discrete_scatter(X_test[:,0], X_test[:,1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, markers='o', ax=ax)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
cbar = plt.colorbar(scores_image, ax=axes.tolist())
axes[0].legend(["Test class 0", "Test class 1", "Train class 0", "Train class 1"], ncol=4, loc=(.1, 1.1))
plt.show()
    #마찬가지로 첫 번째 그림은 decision boundary,
    #두 번째 그림은 predicted probabilities 이다.
    #앞의 방식보다 훨씬 더 구분이 뚜렷하다.
    #scikit-learn 홈페이지에 가면 많은 모델들의 비교 그림이 나와있다.
    #참고하길 바람!



'''
                Uncertainty in Multiclass Classification
'''

#이제까지 이항 분류 모형에만 적용해봤는데, 멀티클래스에도 적용할 수 있다.
#아이리스 예제를 살펴보자.


from sklearn.datasets import load_iris

iris=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

gbrt=GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(X_train, y_train)


#1. decision_function
print("\n\n\nMulticlass Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))
    #다중클래스에서는 결정 함수가 (n_samples, n_classes) 모양을 띠고 있다.
    #각 클래스 별로 점수가 높으면 더 가능성이 큰 것이다.

print("\n\nArgmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
    #각 값에서 최대값인 인덱스와 predict 값이 같은 것을 알 수 있다.
    # = 즉, 값이 높은 것으로 predict 한다.




#2. predict_proba
print("\n\n\nMulticlass predict proba shape: {}".format(gbrt.predict_proba(X_test).shape))
print("predict proba:\n{}".format(gbrt.predict_proba(X_test)[:6]))

print("\n\nArgmax of predict proba:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))
    #예상한 값과 같다.



#요약하자면, 두 방법 모두 (n_samples, n_classes) 형태를 띤다.
#   결정함수의 이항분류의 경우만 예외이다! (n_samples, )형태
#이렇게 수치로 나오는 것들은 argmax 함수를 사용하여 라벨을 복원할 수 있지만,
#문자열이거나 규칙적이지 않은 숫자들은 그렇지 않을 수 있다.
#그럴 때는 앞서 배운 classes_ 를 사용하여 복원하자!



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

# represent each target by its class name in the iris dataset
named_target = iris.target_names[y_train]
logreg.fit(X_train, named_target)
print("unique classes in training data: {}".format(logreg.classes_))
print("predictions: {}".format(logreg.predict(X_test)[:10]))
argmax_dec_func = np.argmax(logreg.decision_function(X_test),axis=1)
print("argmax of decision function: {}".format(argmax_dec_func[:10]))
print("argmax combined with classes_: {}".format(logreg.classes_[argmax_dec_func][:10]))
    #argmax와 classes_ 두 가지 방법을 사용하여 라벨을 복원한 예제
