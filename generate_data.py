import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DataGenerator:
    def __init__(self, n_samples=10, n_features=10, collinearity_coef=0, is_normalized=False, is_real=False, ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.collinearity_coef = collinearity_coef
        self.is_normalized = is_normalized
        self.is_real = is_real

    def __nullspace(self, A, atol=1e-13, rtol=0):
        A = np.atleast_2d(A)
        u, s, vh = np.linalg.svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

    def generate(self, param):
        if self.is_real:
            pass
        else:
            X = np.zeros((self.n_samples, self.n_features))
            if param["type"] == "rand":
                n_samples = self.n_samples
                n_features = self.n_features
                y = np.random.randint(1.5 * n_samples, size=(n_samples,))
                X[:, :n_features - 1] = np.random.rand(n_samples, n_features - 1)
                X[:, n_features - 1] = y + 0.01 * np.random.randn(n_samples)
                if self.is_normalized:
                    y = y / np.linalg.norm(y)
                    col_norm = np.sqrt(np.sum(X ** 2, axis=0))
                    X = X / col_norm
                return X, y
            elif param["type"] == "inadeq":
                pass
            elif param["type"] == "redund_correl":
                y = np.random.randint(1.5 * self.n_samples, size=(self.n_samples,))
                X_ort_y = self.__nullspace(y)
                k = self.collinearity_coef
                X = k * np.repeat(y.reshape(-1, 1), repeats=self.n_features, axis=1) + \
                    (1 - k) * X_ort_y[:, np.random.randint(X_ort_y.shape[1], size=(self.n_features,))]
                if self.is_normalized:
                    y = y / np.linalg.norm(y)
                    col_norm = np.sqrt(np.sum(X ** 2, axis=0))
                    X = X / col_norm
                return X, y
            elif param["type"] == "adeq_correl":
                n_ort_features = int(np.floor(param["ratio_ort_features"] * self.n_features))
                print("Number of orthogonal features = {0}".format(n_ort_features))
                y = np.random.randint(1.5 * self.n_samples, size=(self.n_samples,))
                y = y / np.linalg.norm(y)
                vec1 = np.zeros((self.n_samples,))
                vec2 = np.zeros((self.n_samples,))
                vec1[0::2] = y[0::2]
                vec2[1::2] = y[1::2]
                X[:, 0] = vec1
                X[:, 1] = vec2
                if n_ort_features < 3:
                    X = X[:, :2]
                else:
                    k = self.collinearity_coef
                    X_ort = self.__nullspace(X[:, :2].T)
                    perm_idx = np.random.randint(X_ort.shape[1], size=(n_ort_features - 2,))
                    X[:, 2:n_ort_features] = X_ort[:, perm_idx]
                    col_norm = np.sqrt(np.sum(X[:, :n_ort_features] ** 2, axis=0))
                    X[:, :n_ort_features] = X[:, :n_ort_features] / col_norm
                    y = X[:, 0] + X[:, 1]
                    n_col_features = self.n_features - n_ort_features
                    print('Number of collinear features = {0}'.format(n_col_features))
                    n_col_feat_per_ort_feat = int(np.floor(n_col_features / n_ort_features))
                    print('Number of collinear features per orthogonal feature = {0}'.format(n_col_feat_per_ort_feat))
                    first_idx = n_ort_features
                    for i in list(range(n_ort_features)):
                        last_idx = first_idx + n_col_feat_per_ort_feat
                        X_ort_current_feat = self.__nullspace(X[:, i])
                        X_ort_current_feat = X_ort_current_feat[:, :n_col_feat_per_ort_feat]
                        #                         X_ort_current_feat = X_ort_current_feat[:, np.random.randint(X_ort_current_feat.shape[1],
                        #                                                                                      size=(n_col_feat_per_ort_feat,))]
                        X[:, first_idx:last_idx] = (1 - k) * X_ort_current_feat + \
                                                   k * np.repeat(X[:, i].reshape(-1, 1),
                                                                 repeats=n_col_feat_per_ort_feat, axis=1)
                        first_idx = last_idx
                if self.is_normalized:
                    col_norm = np.sqrt(np.sum(X ** 2, axis=0))
                    X[:, 2:] = X[:, 2:] / col_norm
                return X, y
def return_dataset(dataset_name,n_samples=2000,n_features=30):
    if dataset_name=='protein':
        casp=pd.read_csv('datasets/CASP.csv')
        X=casp.iloc[:,1:]
        y=casp.iloc[:,0].values
        X=MinMaxScaler().fit_transform(X)
    elif dataset_name=='credit':
        credit=pd.read_csv('datasets/UCI_Credit_Card.csv')
        credit['SEX']=credit['SEX'].replace({2:1,1:0})

        credit=pd.concat([credit,pd.get_dummies(credit['EDUCATION'],prefix='education')],axis=1)

        credit=pd.concat([credit,pd.get_dummies(credit['MARRIAGE'],prefix='marriage')],axis=1)

        y=credit['default.payment.next.month'].values

        X=credit.drop('default.payment.next.month',axis=1)

        X=MinMaxScaler().fit_transform(X)
    elif dataset_name=='wine':
        wine=pd.read_csv('datasets/winequality-white.csv',sep=';')
        y=wine['quality'].values
        X=wine.drop('quality',axis=1)
        X=MinMaxScaler().fit_transform(X)
    elif dataset_name=='airbnb':
        X=pd.read_csv('datasets/train.csv')
        y=pd.read_csv('datasets/y_train.csv',names=['target'])['target'].values
        X=MinMaxScaler().fit_transform(X)
    elif dataset_name=='synthetic':
        is_normalized = False
        collinearity_coef = 0.7
        is_real = False
        dg = DataGenerator(n_samples, n_features, collinearity_coef, is_normalized, is_real)
        rand_par = {"type": "rand"}
        inadeq_correl_par = {}
        redund_correl_par = {"type": "redund_correl"}
        adeq_correl_par = {"type": "adeq_correl", "ratio_ort_features": 0.2}
        X, y = dg.generate(adeq_correl_par)
    return X,y