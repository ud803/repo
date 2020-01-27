'''
                    Neural Networks (Deep Learning)
'''


'''
뉴럴 네트워크 (신경망)으로 알려진 알고리즘은 "딥 러닝"이라는 이름으로 최근 다시 떠오르고 있다. 딥 러닝이 유용한 것은 사실이지만, 딥 러닝 알고리즘은 특정한 사용을 위해서 아주 세밀하게 만들어진다.

이 장에서는 비교적 간단한 방법인 multilayer perceptrons을 사용해 분류와 회귀를 할 것이다. 이는 심화된 딥 러닝을 위한 초석으로 삼기에 적합하다.

MLP(Multilayer perceptrons - 다층인식자)는 "(vanilla) feed-forward neural network" 혹은 그냥 "neural network"라고도 알려져 있다.
'''

import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y= make_moons(n_samples =100, noise=0.25, random_state=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

where=input("1 for start, 2 for parameters, 3 for breast cancer : ")
if(where=="1") :

#1. The neural network model
    #MLP는 다수의 처리(processing)과정을 거치는 선형 모델의 일반화라고 할 수 있다.
    # y = w[0]*x[0] + ... + w[p]*x[p] + b
    # 위 식에서, y는 x와 w의 선형 결합이다.

    #MLP에서, 가중합(weighted sum)을 계산하는 과정이 여러번 반복된다.
    #처음에는 중간 처리 단계를 나타내는 "hidden units"를 가중합으로 계산한다.
    #그리고 이것들을 가지고 다시 가중합을 하여 최종 결과를 만들어낸다.
    #사실 수학적으로 가중합을 연속적으로 하는 것은 한 번 계산하는 것과 같다.
    #따라서 여기에 마지막 한 단계를 더 가야하는데,
    #비선형 함수를 결과에 적용시키는 것이다.
    # relu 이전의 방식이 0과 1 사이의 값을 가졌다. hidden layer가 많아질수록 0에 수렴하게 되어서
    # 0이상의 값만 사용하게 되었다.
    #   1) rectified linear unit (relu)
    #       0 미만의 값을 없앤다.
    #   2) tangens hyperbolicius (tanh)
    #       최소값을 -1로, 최대값을 +1로 설정하여 근사하도록 한다.
    #이 두 가지 작업은 신경망이 선형 모델보다 훨씬 더 복잡한 모델을 배우도록 해준다.



    line=np.linspace(-3, 3, 100)
    plt.plot(line, np.tanh(line), label='tanh')
    plt.plot(line, np.maximum(line,0), label='relu')
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("relu(x), tanh(x)")
    plt.show()
        #즉, tanh를 이용하여 이 과정을 수식으로 표현하면 다음과 같다.
        #h[0]=tanh(w[0,0]*x[0] + w[1,0]*x[1] +... )
        #h[1]=...
        #...
        #y = v[0]*h[0] + ...
        # w는 input x와 hidden layer h사이의 가중치이고
        # v는 hidden layer h와 결과값 y사이의 가중치이다.
        # v와 w는 데이터로부터 학습되고, x는 input값, y는 계산된 output값이다.
        # h는 중간 계산을 통해 얻어진다.

#여기서 사용자가 설정하는 중요한 값은 hidden "node"의 숫자이다.
#아주 작은 데이터에는 10개, 아주 큰 데이터에는 10,000개까지 설정할 수 있다.
#또한 hidden layer층을 한 개 더 (수평적으로) 추가하는 것도 가능하다.
#이런 방식으로 hidden layer를 늘려나가는 것이 "딥러닝"의 아이디어가 되었다.



#2. Tuning neural networks
    #two_moon 데이터셋에 MLP를 적용해보자.




    mlp = MLPClassifier(solver='lbfgs',random_state=0).fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

        #매우 비선형이고 비교적 부드러운 decision boundary가 탄생하였다.
        #solver='lbfgs'는 나중에 설명
        #디폴트 값으로 100개의 hidden node를 사용한 결과이다.
        #작은 데이터셋임에도 불구하고 100개는 조금 크다.
        #숫자를 줄여보자. (= 모델의 복잡도를 낮춰보자.)
        #simple = generalization = 부드러운 = 간단한

    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        # hidden node 10으로 줄이니까, 더 지저분해졌다.
        # 디폴트 비선형 함수는 relu인데, hidden layer가 1개라면
        # 이것은 결정 함수가 10개의 직선 조각으로 만들어졌다는 것을 의미한다.(relu=직선)
        # 부드러운 경계를 원한다면
        #   1) hidden unit을 늘리거나 (앞에서 한 것처럼)
        #   2) 두 번째 hidden layer를 추가하거나
        #   3) tanh 비선형을 사용할 수 있다.


    #두 번째 layer 추가
    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10,10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()


    #tanh 비선형 + two hidden layers
    mlp = MLPClassifier(solver='lbfgs', activation='tanh',random_state=0, hidden_layer_sizes=[10,10])
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()
        #옵션을 추가할수록 점점 더 부드러워지는 것을 알 수 있다.

        #또한, 뉴럴 네트워크의 복잡도를 통제하는 방법으로 "l2 penalty"가 있다.
        #가중치를 0으로 낮추는 것인데, 리지 회귀와 선형 분류에서 썼던 방법이다.
        #여기에 해당하는 파라미터는 alpha인데, 아주 낮은 값으로 (낮은 정규화) 설정되어있다.

        #서로 다른 alpha값에 따른 변화를 아래서 살펴보자.

elif(where=="2") :

    fig, axes = plt.subplots(2, 4, figsize=(20,8))
    for axx, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
            mlp.fit(X_train, y_train)
            mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, ax=ax)
            ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
    plt.show()
        #모델이 간단할수록 더 부드러워진다.
        #이제 우리는 뉴럴 네트워크의 복잡도를 3가지 변수가 통제한다는 것을 알았다.
        # 1) hidden layer의 수
        # 2) 각 layer에 있는 hidden unit의 수
        # 3) 정규화 정도(alpha)


#뉴럴 네트워크에서, 가중값은 학습이 시작되기 전에 "임의로" 정해진다.
#그리고 이 임의의 배분은 학습 모델에 영향을 미친다.
#즉, 같은 파라미터를 쓰더라도 랜덤 시드에 따라 그 모델이 바뀐다는 점이다.
#네트워크가 크다면, 그 임의성이 별 의미가 없겠지만, 작은 네트워크에서는 유의미함을 주의!
#아래에서 서로 다른 랜덤 시드에 따른 차이점을 살펴보자.

    fig, axes = plt.subplots(2, 4, figsize=(20,8))
    for i, ax in enumerate(axes.ravel()):
        mlp=MLPClassifier(solver='lbfgs', random_state=i, hidden_layer_sizes=[100,100])
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
        mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train, ax=ax)
    plt.show()


#이제 실생활 데이터에 적용시켜 보자.

elif(where=="3") :

    from sklearn.datasets import load_breast_cancer

    cancer = load_breast_cancer()
    print("Cancer data per-feature maxima:\n{}".format(cancer.data.max(axis=0)))

    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    mlp=MLPClassifier(random_state=42)
    mlp.fit(X_train,y_train)

    print("\nMLP Score\n")
    print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
    print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
        #0.91 0.88
        #점수가 좋지 않은데, 이는 뉴럴 네트워크도 SVC처럼 데이터의 스케일링이 필요하기 때문이다.
        #뉴럴 네트워크에서는 평균 0, 분산 1의 데이터를 이상적으로 본다.
        #여기서는 직접 그 값을 만들고, Ch3 이후부터는 StandardScaler를 사용한다.


    #compute the mean value per feature on the training set
    mean_on_train = X_train.mean(axis=0)
    #compute the standard deviations of each feature on the training set
    std_on_train = X_train.std(axis=0)

    #subtract the mean, and scale by inverse standard deviations
    #afterward, mean=0 and std = 1
    X_train_scaled = (X_train - mean_on_train)/std_on_train

    X_test_scaled = (X_test- mean_on_train)/std_on_train

    mlp=MLPClassifier(random_state=0)
    mlp.fit(X_train_scaled, y_train)

    print("\nMLP Scaled Score\n")
    print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
        #0.991 0.965
        #결과는 매우 좋아졌지만, 경고 문구가 뜬다.
        #이는 모델을 학습하는데 쓰이는 'adam' 알고리즘 때문인데, 반복 수를 늘려야한다.


    mlp=MLPClassifier(max_iter=1000, random_state=0)
    mlp.fit(X_train_scaled, y_train)

    print("\nMLP Scaled Score (Iter=1000)\n")
    print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
        #0.993 0.972
        #결과 값 사이에 어느정도 차이가 있으므로, 모델의 복잡도를 낮춰보자.
        #alpha는 0.0001에서 1로 증가시킬 것이다!


    mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
    mlp.fit(X_train_scaled, y_train)

    print("\nMLP Scaled Score (Iter=1000, Alpha=1)")
    print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


#뉴럴 네트워크가 무엇을 학습했는지 분석하는 것은 가능하지만, 다른 방법들에 비해 더 까다롭다
#한 가지 방법은 모델에 쓰인 가중치들을 보는 것이다.
#다음 그림을 살펴보자.

    plt.figure(figsize=(20,5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()
    plt.show()
        #행은 30개의 input feature를 의미하고,
        #열은 100개의 hidden unit을 의미한다.
        #밝은 색은 큰 양의 값을, 어두운 색은 음의 값을 의미한다.
        #이와 반대로 hidden layer를 output layer에 연결하여 시각화할 수 있지만,
        #더 해석하기가 힘들다.

#MLPClassifier와 MLPRegressor가 손쉬운 인터페이스를 제공하지만, 이들은
#뉴럴 네트워크에서 아주 일부분만을 보여준다.
#더 유연하고 방대한 뉴럴 네트워크를 경험하고 싶다면 scikit-learn에서 벗어나
#keras, lasagna, tensor-flow를 경험해보아야 한다.
#이들은 GPU의 사용을 허용해주는데, 이는 10배~100배가량 더 빠른 속도를 내어 큰 데이터셋에 필수적이다.




#3. Strengths & Weaknesses
    #뉴럴 네트워크는 최첨단 모델로서 머신 러닝의 많은 분야에 적용되고 있다.
    #가장 큰 장점은 큰 데이터에 포함된 정보를 찾아내고, 놀랍도록 복잡한 모델을 만든다는 것이다.
    #충분한 연산시간과 데이터가 주어지고, 파라미터가 잘 조율된다면 뉴럴 네트워크는 다른 알고리즘들을 성능에서 압도한다.
    #하지만 뉴럴 네트워크는 학습하는 데 오래 걸린다.
    #또한 데이터의 전처리가 필요하고, SVM처럼 비슷한 종류(homogeneous)의 데이터끼리 있을 때 좋다.
    #우리가 이 장에서 배운 뉴럴 네트워크 모델은 빙산의 일각일 뿐이다.


#4. Estimating complexity in neural networks.
    #가장 중요한 파라미터는
    #   1) number of layer
    #   2) number of hidden units per layer
    #   3) alpha
    # 만약 100 feature가 있고 100 hidden unit이 있다면, 100*100=10000 weight이 있다. (input과 first hidden layer 사이에)
    # 추가로 100*1 weight도 있다. (hidden layer와 output 사이에)

    #파라미터를 조율하는 방법은
    #   1)먼저 아주 큰 네트워크를 만들어서 "overfit"하도록 만든 뒤,
    #   2)네트워크를 줄이거나 alpha를 늘려서 generalization하도록 만드는 것이다.

    #이 장에서 우리는 뉴럴 네트워크의 정의와 개념에 대해서 배웠지만
    #어떻게 동작하는지는 배우지 않았다.
    #사실 'algorithm'이라는 키를 통해 다른 알고리즘을 사용할 수 있다.
    #디폴트인 'adam'은 대부분의 환경에서 잘 동작하지만 스케일링에 민감하다.
    #'l-bfgs'는 정교하지만 큰 데이터셋에서는 시간이 오래 걸린다.
    #'sgd'는 많은 딥러닝 연구자들이 쓰는 것으로, 많은 추가적인 파라미터를 제공한다.
    #MLP로 시작할 때는, adam과 l-bfgs를 쓰는 것을 추천한다.
