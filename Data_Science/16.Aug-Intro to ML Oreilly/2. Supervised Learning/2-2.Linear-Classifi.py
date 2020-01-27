where=input("2 for binary, 3 for multi : ")
if(where=='2'):


    '''
                        2. Linear models for Classification




                            Binary Classification

        # 2항 분류에서는 다음과 같은 공식을 사용한다
        # y= w[0]*x[0] + w[1]*x[1] + ... + w[p]*x[p] + b > 0
        # 회귀 분석에 쓴 공식과 비슷하지만, 차이점은 여기서는 0보다 큰 값을 다룬다.
        # 만약 예측값이 0보다 작을 경우, 우리는 그 클래스를 -1이라고 예측한다.
        # 만약 예측값이 0보다 큰 경우, 그 클래스를 +1이라고 예측한다.
    '''


    '''
    회귀 분석에서, 예측값인 y는 feature의 선형 결합 함수였다 (즉, 직선이나, 평면이냐, 하이퍼플레인)
    분류에서는, "decision boundary"라는 것이 인풋에 대한 선형 함수이다.
        다시 말해서, (2항) 분류란 직선이나 평면, 하이퍼플레인을 사용하여 두 계층을 나누는 것이다.
    회귀 분석과 마찬가지로 방법들이 있지만, 각자는 아래와 같은 이유로 조금씩 다르다
        1) 특정 w와 b의 값들이 얼마나 트레이닝 셋에 잘 맞는지
        2) 어떤 종류의 정규화를 사용하는지
    각 알고리즘은 "트레이닝 셋에 잘 맞는것"을 측정하기 위해 다른 방법을 사용한다.
    기술적 이유로, 잘못된 예측을 최소화하기 위해 w와 b를 조정하는 것은 불가능하다.

    우리는 "Logistic Regression"과 "linear SVM"을 살펴볼 것이다. Logistic Regression은 그 이름과는 달리 classification algorithm이다.
    '''


    #2. Logistic Regression & Linear svm 큰 그림

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import matplotlib.pyplot as plt
    import mglearn

    X, y = mglearn.datasets.make_forge()

    fig, axes = plt.subplots(1,2, figsize=(10,3))

    for model, ax in zip([LinearSVC(), LogisticRegression()], axes) :
        clf = model.fit(X,y)
        mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
        ax.set_title("{}".format(clf.__class__.__name__))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Featrue 1")
    axes[0].legend()
    plt.show()
        # 두 모델 모두 L2 정규화를 사용한다.
        # 정규화의 정도는 'C'라는 파라미터를 사용하는데,
        #   C가 클수록 정규화의 정도가 낮고, 트레이닝셋에 지향된다.
        #   C가 작을수록 정규화의 정도가 크고, 더 0에 가까운 계수(w)를 찾으려고 한다.

        # C의 값이 작으면 알고리즘은 데이터셋의 "대다수"에 맞추려고 하는 경향이 있다.
        # C의 값이 크면 각각의 데이터가 정확하게 분류되도록 강조한다.
    mglearn.plots.plot_linear_svc_regularization()
    plt.show()
        # 작은 C는 높은 정규화를 의미한다.
        # 조금 더 큰 C는 잘못 분류된 두 샘플에 더 집중한다.
        # 매우 큰 C는 모든 점을 정확히 분류하려고 한다. 하지만 큰 그림을 못 볼 수 있다.
        #       즉, C가 커지면 overfitting할 수 있다.
        #


    '''
    낮은 차원에서, 선형 분류는 매우 제한적으로 보이지만, 고차원으로 갈수록 (feature가 많아질수록) 선형 분류는 매우 강력해진다. 따라서 overfitting을 막는 문제가 매우 중요해진다.
    '''


    #3. Logistic Regression (심화)

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import  load_breast_cancer
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    logreg = LogisticRegression().fit(X_train,y_train)

    print("\nLOGREG score\n")
    print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg.score(X_test,y_test)))
        # 결과는 0.955 0.958로서 매우 높지만,
        # 두 점수가 비슷한 것으로 보아 underfitting의 가능성이 있다.
        # C = 100으로 늘려보자.

    logreg100 = LogisticRegression(C=100).fit(X_train,y_train)
    print("\nLOGREG100 score\n")
    print("Training set score: {:.3f}".format(logreg100.score(X_train,y_train)))
    print("Test set score: {:.3f}".format(logreg100.score(X_test,y_test)))
        # 결과는 0.977 0.965로 더 좋아졌다. 우리의 예상이 맞았다.

    logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)
    print("\nLOGREG0.01 score\n")
    print("Training set score: {:.3f}".format(logreg001.score(X_train,y_train)))
    print("Test set score: {:.3f}".format(logreg001.score(X_test,y_test)))
        # 결과는 0.934 0.93
        # 이미 underfit한 모델에서 더 왼쪽으로(그림2-1기준) 가니까 정확도가 낮아졌다.
    # 마지막으로, 각 C에 따른 계수(w)들을 그림으로 살펴볼 것이다.

    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()
        # C가 작을수록 계수들이 0에 가까워지지만 절대 0은 되지 않는다!
        # 도표를 해석할 때 주의해야 한다. 왜? (이해못함)


    '''
    더 인터프리터블한 결과를 원한다면, L1 정규화를 사용하면 된다.
    '''

    #3-2. Logistic Regression using L1

    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1=LogisticRegression(C=C, penalty="l1").fit(X_train,y_train)
        print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, lr_l1.score(X_test,y_test)))
        plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
    plt.hlines(0, 0, cancer.data.shape[1])
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")

    plt.ylim(-5,5)
    plt.legend(loc=3)
    plt.show()
        # 선형 회귀와 마찬가지로, 모델간의 차이는 "penalty parameter"이다.
        # 이는 모델이 모든 feature를 사용할 것인지, 그 중 일부만을 사용할 것인지 결정한다.


else :
'''
                        Multiclass Classification
'''


'''
많은 선형 분류 모델은 이항 분류만을 위한 것들이 많고, 그것들은 멀티클래스 분류는 지원하지 않는다. (Logistic Regression은 예외) 이항 분류를 멀티 클래스로 확장하는 알고리즘은 "one- vs. -rest"이다.
이 방법에서, 이항 분류는 각 클래스마다 지정되어 각 클래스를 나머지 다른 것들로부터 분류하는 방식으로 진행된다. (결국 클래스 개수만큼 바이너리 모델이 생김)
예측을 하기 위해서, 훈련된 모든 바이너리 모형이 test 셋에 대해 실행되고, 가장 높은 점수를 얻은 모형이 "승리"하게 되어 그 클래스의 라벨이 리턴된다.
'''


    #1. one- vs. rest- approach

    from sklearn.datasets import make_blobs
    import mglearn
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import numpy as np

    # 데이터셋 생성
    X, y = make_blobs(random_state=42)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(["Class 0", "Class 1", "Class 2"])
    plt.show()


    # training
    linear_svm = LinearSVC().fit(X,y)
        # LinearSVC는 2개 이상의 클래스가 감지되면 multiclass 옵션을 실행한다
        # default가 one-vs-rest 방식
        # 다른 방식도 지원함
    print("Coefficient shape: ", linear_svm.coef_.shape)
    print("Intercept shape: ", linear_svm.intercept_.shape)
        # (3,2) (3,)
        # 3개의 클래스, 각각 2개의 feature
        # intercept는 각 클래스의 intercept, 따라서 3개

    # visualize
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    line = np.linspace(-15,15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
        plt.plot(line, -(line*coef[0]+intercept)/coef[1], c=color)
    plt.ylim(-10,15)
    plt.xlim(-10,8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.show()
        # 선에 의해 분류될 수 있으면 분류가 된다.
        # 중간의 삼각형같은 사각지대가 생기면 그 중 가장 값이 큰 것으로 라벨링.
        #   여기서 값이 큰 것은 가장 가까운 직선의 클래스를 의미.


# 아래는 각 지점이 어디로 분류되는지 보여주는 예측도

    mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
    mglearn.discrete_scatter(X[:,0], X[:,1], y)
    line = np.linspace(-15,15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b','r', 'g']):
        plt.plot(line, -(line*coef[0]+intercept)/coef[1], c=color)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(0.9, 0.3))
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()





'''Linear 마무리

1)
Linear의 주요 파라미터는
    Regression에서는 alpha,
    Classification에서는 C이다.
Alpha가 크거나, C가 작으면 단순한 모델을 의미한다. 보통 이 숫자들은 로그 단위로 쓰인다.


2)
L1 정규화와 L2 정규화 중 적절한 것을 사용할 줄 알아야 한다.
여러 feature중 몇 개만 중요한 것 같으면 L1을 써야한다. 또한 해석을 용이하게 하기 위해서도 L1이 더 좋다.


3)
선형 모델은 훈련과 예측이 빠르다. 또한 직관적으로 이해하기가 쉽다. 하지만, 데이터셋의 feature들 간의 상관 관계가 높다면 계수들의 의미를 파악하기가 힘들다.

선형 모델은 샘플 수에 비해 feature의 수가 많을 때 성능이 좋다. 또한 매우 큰 데이터셋에도 좋은데, 이는 단순히 다른 모델들이 큰 데이터셋에 적합하지 않기 때문이다.

하지만, 낮은 차원의 feature에서는, 다른 모델들이 선형 모델에 비해 generalization 성능이 더 좋다!

'''
