import numpy as np
from tqdm import tqdm


class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        m=len(X)
        oup=np.zeros(m)
        for i in range(m):
            lst=[]
            for j in range(len(self.X)):
                d=np.sum((self.X[j]-X[i])**2)
                lst.append(d)
            nearest = np.argsort(lst)
            topK_y = [self.y[kk] for kk in nearest[:self.k]]
            oup[i]=max(set(topK_y), key=topK_y.count)
            if(i%100==0):
                print(i)
        return oup
            


        
        # raise NotImplementedError
        ...

        # End of todo