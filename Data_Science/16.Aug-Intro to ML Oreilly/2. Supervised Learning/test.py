import mglearn
from sklearn.datasets import make_blobs

X, y =make_blobs(centers=4, random_state=8)
y=y%2

print("\nX",X)
print("\nY",y)

print("\nX[:,0]", X[:,0])
print("\nX[:,1]", X[:,1])

print("\n",X[:,1:])
