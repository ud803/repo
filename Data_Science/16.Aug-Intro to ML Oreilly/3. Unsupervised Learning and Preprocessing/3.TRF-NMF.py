'''
                Non-Negative Matrix Factorization (NMF)
'''

'''
NMF는 유용한 feature를 추출하기 좋은 또 다른 비지도 학습 알고리즘이다. PCA와 비슷하게 작동하고, 차원 감소를 위해 사용한다. PCA에서와 마찬가지로, 각 데이터를 컴포넌트들의 가중합으로 표현하지만, 다른 점이 있다면 NMF에서 그 컴포넌트와 계수들이 모두 0보다 크거나 같아야 한다는 점이다.

결과적으로, 이 방법은 각 feature가 음이 아닌 값을 가지는 경우에만 적용될 수 있다. 이 방식은 특히 독립적인 소스들의 결합으로 나타내어지는 데이터들에게 좋다. 그 예로는 여러 사람이 동시에 말하는 오디오 트랙이나 많은 악기가 동시에 연주되는 음악 트랙이 있다.

전반적으로, NMF는 PCA보다 더 해석하기 쉽다. PCA에서는 음의 값을 지닌 컴포넌트와 계수들이 해석을 어렵게 하기 때문이다.
'''

import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

where=input("1 for faces, 2 for synthetic data : ")
if(where=="1"):
#1. Applying NMF to synthetic data
    #우리의 데이터가 양의 방향에 있어야 한다.
    #즉, 원점에 비추어 데이터가 어느 방향을 향하는지도 중요하다는 말이다.

    mglearn.plots.plot_nmf_illustration()
    plt.show()
        #컴포넌트가 두 개일 때, 모든 데이터는 두 벡터의 합으로 나타내어진다.
        #따라서 feature의 개수만큼 컴포넌트가 존재한다면, 완벽하게 데이터를 복원할 수 있다.
        #컴포넌트가 한 개일 때, 컴포넌트는 평균을 가리키는 벡터이다.
        #PCA와 달리, 컴포넌트의 수를 줄이면 몇몇 방향이 사라질 뿐 아니라 완전히 새로운 컴포넌트를 만들어 낸다.
        #(즉, 컴포넌트2개일 때와 1개일 때 서로 벡터가 다르다는 말, PCA에서는 같았다.)
        #NMF는 랜덤 요소가 있으므로, 그때그때 달라질 수 있다.


#2. Applying NMF to face images
    #NMF의 주 파라미터는 추출할 컴포넌트의 개수이다.
    #보통, input feature보다 그 수가 적은데, 그렇지 않으면 데이터가 각 픽셀을 별개의 컴포넌트로 만들어버릴 수 있기 때문이다.


    people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] =1

    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255

    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

#mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
#plt.show()
#안됨!!

    nmf = NMF(n_components=15, random_state=0)
    nmf.fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)

    fix, axes = plt.subplots(3, 5, figsize=(15,12), subplot_kw={'xticks' : (), 'yticks': ()})

    for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape))
        ax.set_title("{}.component".format(i))
    plt.show()
        #먼저 NMF를 통해 15개의 컴포넌트만 추출하였다.
        #이는 확실히 PCA보다 얼굴들을 더 잘 대표한다.
        #컴포넌트3은 오른쪽을 보고있고, 컴포넌트7은 왼쪽을 보고있다.
        #이제 이 컴포넌트들의 특성을 잘 살리는 이미지를 살펴보자.



# sort by 3rd component, plot first 10 images
    compn=3
    inds = np.argsort(X_train_nmf[:,compn])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(15,8), subplot_kw={'xticks':(), 'yticks': ()})
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
    plt.show()


# sort by 7th component, plot first 10 images
    compn=7
    inds = np.argsort(X_train_nmf[:,compn])[::-1]
    fig, axes = plt.subplots(2,5,figsize=(15,8), subplot_kw={'xticks':(), 'yticks': ()})
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape))
    plt.show()
        #우리의 예상대로, 각 컴포넌트의 계수가 높은 상위 10개의 데이터를 보여준다.
        #이렇게 패턴을 추출하는 것은 '더하는 구조', 즉 오디오, 유전자, 텍스트 데이터에 유용하다.
        #이제 가상의 데이터를 만들어보자.

elif(where=='2') :



    S = mglearn.datasets.make_signals()
    plt.figure(figsize=(6,1))
    plt.plot(S,'-')
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.show()
        #우리는 여기서 원본 신호를 골라내고 싶다.
        #이를 위해, 이 혼합 신호를 측정할 100개의 장치가 있다고 가정하자.

#mix data into a 100-dim state
    A = np.random.RandomState(0).uniform(size=(100,3))
    X = np.dot(S, A.T)
    print("Shape of measurements: {}".format(X.shape))
        #2000개의 100차원(feature가 100개인) 데이터

    nmf = NMF(n_components=3, random_state=42)
    S_ = nmf.fit_transform(X)
    print("Recovered signal shape: {}".format(S_.shape))
        #100개의 feature 중에서 3개의 컴포넌트만을 추출하였다.
        #nmf는 임의성을 띠므로 random state 파라미터가 들어감.


#비교를 위해, PCA도 적용하여보자.
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)

    models = [X,S,S_,H]
    names = ['Observations (first three measurements)', 'True sources', 'NMF recovered signals', 'PCA recovered signals']
    fig, axes = plt.subplots(4, figsize=(8,4), gridspec_kw={'hspace':.5}, subplot_kw={'xticks':(), 'yticks':()})

    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:,:3],'-')
    plt.show()
        #NMF는 원본을 잘 찾아내었다.
        #PCA는 첫 번째 컴포넌트로 대다수의 데이터를 설명하려다가 실패했다.
        #NMF가 만든 컴포넌트는 자연적인 순서가 없다는 것에 주의하자.
        #PCA는 그 중요도에 따라 순서가 있었다.
