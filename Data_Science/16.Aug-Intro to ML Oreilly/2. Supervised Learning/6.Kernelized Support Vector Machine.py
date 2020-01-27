'''
                Kernelized Support Vector Machines
'''

'''
우리가 다룰 내용은 앞서 선형 모델에서 보았던 "SVC(Support Vector Classifier)"와 유사하지만, 단순히 하이퍼플레인으로 정의되지 않는 조금 더 복잡한 모델을 지원한다.
'''

import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

where=input("1 for basic idea, 2 for kernel trick, 3 for scaling : ")
if(where=="1") :


#1. Basic Idea
#선형 모델은 낮은 차원에서 매우 제한적이다. 여기에 feature를 추가하면 모델을 더 유연하게 만들 수 있다.


    X,y = make_blobs(centers=4, random_state=8)
    y = y%2

    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        #이런 데이터셋에는 직선 하나로 나누는 선형 모델은 적합하지 않을 것이다.

    from sklearn.svm import LinearSVC
    linear_svm = LinearSVC().fit(X,y)

    mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        #결과를 보면 적합하지 않다!


#이제 input feature를 확장시켜보자.
#feature1 ** 2 (즉, 두번째 feature의 제곱)을 새로운 feature로서 추가하자.
#이제 2차원 평면이 아닌 3차원 공간에서 그림으로 표현할 것이다.


#add the squared first feature
    X_new = np.hstack([X, X[:,1:] **2])

    from mpl_toolkits.mplot3d import Axes3D, Axes3D
    figure = plt.figure()

#visualize in 3D
    ax = Axes3D(figure, elev=-152, azim=-26)

#plot first all the points with y=0, then all with y=1
    mask = y == 0
    ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b', cmap=mglearn.cm2, s=60)
    ax.scatter(X_new[~mask,0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 **2")
    plt.show()
        #이제는 선형 모델을 통해 두 클래스를 분류할 수 있다.


    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    #show linear decision boundary
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    xx = np.linspace(X_new[:,0].min() -2, X_new[:,0].max()+2, 50)
    yy = np.linspace(X_new[:,1].min() -2, X_new[:,1].max()+2, 50)

    XX, YY = np.meshgrid(xx,yy)
    ZZ = (coef[0]*XX + coef[1]*YY + intercept) / -coef[2]
    ax.plot_surface(XX,YY,ZZ, rstride=8, cstride=8, alpha=0.3)

    ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b', cmap=mglearn.cm2, s=60)

    ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r', marker='^', cmap=mglearn.cm2, s=60)
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 **2")
    plt.show()
        #이제 3차원 평면의 선형 모델로 둘을 분류하는 게 가능해졌다.



#기존 feature의 함수로 인해, 선형 SVM모델은 사실상 선형이 아니게 된다.(제곱이 있어서)
#직선보다는, 타원형에 가까워지는데, 다음 그림을 통해 알 수 있다.
    ZZ = YY**2
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        #실제로 Decision boundary가 직선이 아닌 타원형이다.

elif where=="2" :





#2. The Kernel Trick
    #우리는 비선형 feature를 추가하는 것이 선형 모델을 강력하게 만든다는 것을 보았다.
    #그렇다면 무슨 기준으로 feature에 변화를 주는 것일까?
    #몇가지 수학적 트릭이 있는데, "kernel trick"이라고 부른다.
    #   1)polynomial kernel : 원래 feature로부터 가능한 다항식을 만들어낸다.
    #   2)radial basis function(RBF) or Gaussian Kernel
    #     :가능한 모든 다항식을 고려하지만, 고차원일수록 중요도가 낮아진다.
    #적용함에 있어서 수학적 디테일은 그렇게 중요하지 않지만, 간단히 살펴보도록 하자.


#3. Understanding SVMs
    #먼저 SVM은 각 트레이닝 데이터가 decision boundary에 미치는 중요도를 학습한다.
    #보통 두 클래스 간의 경계에 있는 데이터가 그러한데, 이들을 support vector라 부른다
    #이 support vector들로부터의 거리를 측정하여, 분류를 한다.
    #   거리 rbf(x1,x2) = exp(r||x1-x2||^2)

    X, y =mglearn.tools.make_handcrafted_dataset()
    svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)
    mglearn.plots.plot_2d_separator(svm,X, eps=.5)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    #plot support vectors
    sv = svm.support_vectors_

    #class labels of support vectors are given by the sign of the dual coef
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        #곡선이 매끄럽게 나오는데, 파라미터인 C와 gamma를 조정해주었기 때문이다.
        #아래서 자세하게 살펴보자.



#4. Tuning SVM parameter
    #gamma는 가우시안 커널의 너비를 조절한다. 즉, 점들이 가까이 있는 정도를 통제한다.
    #C는 정규화 파라미터이고, 각 점의 중요도를 통제한다.
    #파라미터의 변화에 따른 결과값을 살펴보자.

    fig, axes = plt.subplots(3,3,figsize=(15,10))
    for ax, C in zip(axes,[-1, 0, 3]):
        for a, gamma in zip(ax, range(-1,2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

    axes[0,0].legend(["Class 0", "Class 1", "sv class 0", "sv class 1"], ncol=4, loc=(.9,1.2))
    plt.show()
        #낮은 gamma는 가우시안 커널의 반지름이 커진다는 것을 의미한다.
        #   즉, 많은 점들이 가깝다고 간주되는 것이다.
        #낮은 gamma는 decision boundary가 천천히 바뀐다는 것을 의미한다.
        #   이는 낮은 복잡도의 모델을 의미한다.

        #낮은 C는 아주 제한된 모델을 의미한다.
        #   각 데이터의 영향력이 낮아서, 잘못 분류되었음에도 거의 선에 영향을 미치지 못한다.
        #C를 증가시키면, 각 점이 영향력이 커지고, decision boundary가 휘기 시작한다.


elif(where=="3") :
# Breast Cancer 데이터에 적용시켜보자.


    cancer = load_breast_cancer()

    X_train, X_test ,y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    svc=SVC()
    svc.fit(X_train, y_train)

    print("\nSVC score\n")
    print("Accuracy on training set : {:.2f}".format(svc.score(X_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
        #overfit이 심각하다.
        #SVM은 파라미터 설정과 데이터의 크기 조정에 매우 민감하다.
        #특히, 모든 feature가 비슷한 크기 내에서 차이가 나야한다.
        #각 feature의 min, max 값을 그려보자.

    plt.plot(X_train.min(axis=0), 'o', label='min')
    plt.plot(X_train.max(axis=0), '^', label='max')
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.show()
        #각 feature의 크기가 매우 다르다는 것을 알 수 있다.
        #이것은 다른 모델(선형 모델)에게 약간의 문제가 될 수 있지만,
        #kernel SVM에는 아주 치명적인 문제가 된다.
        #이를 해결할 방법을 찾아보자.


#5. Preprocessing data for SVMs
    #각 feature를 rescale하여 비슷한 크기로 만들어주어야 한다.
    #가장 자주 쓰이는 방법은 모든 feature가 0과 1 사이의 값을 갖도록 하는 것이다.
    #챕터 3에서는 'MinMaxScaler'를 사용하겠지만, 여기서는 직접 해보도록 하자.

#compute the min value per feature on the training set
    min_on_training = X_train.min(axis=0)
#compute the range of each feature (max-min) on the training set
    range_on_training = (X_train - min_on_training).max(axis=0)

#subtract the min, and divide by range
#afterward, min=0 and max=1 for each feature

    X_train_scaled = (X_train-min_on_training) / range_on_training

    print("\n")
    print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
    print("Maximum for each feature\n{}".format(X_train_scaled.max(axis=0)))
        #이제 0과 1 사이로 rescaling이 되었다.
        #이 데이터를 가지고 다시 모델에 넣어보자.

    X_test_scaled = (X_test - min_on_training)/ range_on_training

    svc = SVC()
    svc.fit(X_train_scaled, y_train)

    print("\nScaled Score\n")
    print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
        #0.948 0.951
        #데이터를 rescaling 한 것 만으로 엄청난 차이가 있다.
        #두 값이 비슷하므로, underfitting하고 있다. C나 gamma를 늘려 더 복잡하게 만들자.

    svc=SVC(C=1000)
    svc.fit(X_train_scaled, y_train)
    print("\nScaled Score (C=1000)\n")
    print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
        #0.988 0.972
        #C를 증가시켰더니 더 좋은 값이 나왔다.



#6. Strengths & Weaknesses
    #데이터가 낮은 featue를 가지고 있어도 허용이 된다.
    #low~high dimension에서 모두 잘 작동한다.
    #10000개의 샘플정도는 괜찮지만, 10만개를 넘어가면 시간과 메모리 측면에서 안좋다.
    #데이터의 전처리와 파라미터의 설정이 매우 중요하다.
    #   그래서 요즘 추세는 트리 기반 모델(랜덤 포레스트, 그래디언트 부스팅)을 쓰는 것
    #예측값에 대한 설명이 힘들 수 있다.
    #feature 모두가 비슷한 단위로 이루어져 있으면 (pixel, cm..) 적용하기에 좋다.

    #파라미터 C, 커널의 선택, 커널마다의 파라미터를 잘 조정하는게 중요하다.
    #우리는 RBF커널을 사용했지만, 다른 커널도 있다.
    #RBF의 경우 gamma가 큰 경우 복잡도도 크다.
    #C와 gamma가 함께 조정되어야 한다.
