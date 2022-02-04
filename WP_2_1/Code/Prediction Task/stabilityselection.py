import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.construct import random

from scipy import interpolate
import random

from sklearn import linear_model

class Pandas_Standard_Scaler():
    
    def fit(self,X,y=None):
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()
        
        self.mean = X.mean(axis = 0)
        self.std = X.std(axis = 0)
        return self
        
    def transform(self, X):
        X = X.copy()
        if type(X) == pd.core.frame.DataFrame:
            X =X.loc
        X[:,:] -= self.mean
        X[:,:] /= self.std
        return X
    


class Colinearity_Remover():
    def __init__(self, threshold = 0.9):
        self.threshold = threshold
        
    def fit(self,X,y = None):
        if type(X) != pd.core.frame.DataFrame:
            dataset = pd.DataFrame(X)
        else:
            dataset = pd.DataFrame(X.to_numpy())
            
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) >= self.threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
        
        self.feature_mask = [False if i in col_corr else True for i in range(len(corr_matrix.columns)) ]
        
        return self
        
    def transform(self, X):
        if type(X) == pd.core.frame.DataFrame:
            X =X.loc
        return X[:,self.feature_mask]
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)


    
class Stability_Selection():
    
    def __init__(self,
                 num_bootstraps = 100,
                 n_alphas = 100,
                 stability_threshold = 0.6):

        self.num_bootstraps = num_bootstraps
        self.n_alphas = n_alphas
        self.stability_threshold = stability_threshold

    def fit(self, X, y):
        
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()
        if type(y) == pd.core.frame.DataFrame:
            y = y.to_numpy()

        coeff_dict = {}
        min_alphas = []
        max_alphas = []
        
        num_patients = len(X)
        for i in range(self.num_bootstraps):
            subsample_patients = random.sample(range(num_patients), int(num_patients/2))
            X_subsample, y_subsample = X[subsample_patients,:], y[subsample_patients]
            
            alphas, coefs, _ = linear_model.lasso_path(X_subsample, y_subsample, n_alphas = self.n_alphas)

            # we fit interpolation functions f(alpha) = coeff, as the optimal regularization path is different
            # for the different sub samples
            coeff_dict[i] = {}
            alphas = alphas
            for j in range(len(coefs)):
                coeffs = coefs[j,:]
                coeff_dict[i][j] = interpolate.interp1d(alphas, coeffs)
                
            min_alphas.append(alphas.min())
            max_alphas.append(alphas.max())
            
        max_min_alpha = max(min_alphas)
        min_max_alpha = min(max_alphas)
        
        print(max_min_alpha, min_max_alpha)
        
        assert max_min_alpha < min_max_alpha, "The different boostraps have non-compatible alpha ranges. This might be due to outliers in the data or improper scaling of the features. Try a scaling of the features that removes the outliers. Additionally try to lower the alpha_min_ratio (this will lead to longer computation time of the algorithm)."
    
        #generate stability matrix from interpolation functions
        alphas = np.logspace(np.log(max_min_alpha*1.01), np.log(min(max_alphas)*0.99),self.n_alphas,base=np.exp(1))
        stability_path_matrix = 0
        for i in range(self.num_bootstraps):

            coefs_interpolated = np.zeros([X.shape[1],self.n_alphas])

            for j in range(X.shape[1]):
                coefs_interpolated[j,:] = coeff_dict[i][j](alphas)

            # feature acceptance: only accept features that are larger than the median value of the features
            coeffs_smaller_equal_median = (coefs_interpolated <= np.median(coefs_interpolated,axis = 0))
            coef_selected = np.ones(coefs_interpolated.shape)
            coef_selected[coeffs_smaller_equal_median] = 0 

            stability_path_matrix += coef_selected

        stability_path_matrix/=self.num_bootstraps
        self._alphas = alphas
        self._stability_path_matrix = stability_path_matrix
        
        self.feature_mask = stability_path_matrix.max(axis = 1) > self.stability_threshold
        return self

    def transform(self,X):
        
        if type(X) == pd.core.frame.DataFrame:
            X =X.loc
        return X[:,self.feature_mask]
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)

    def plot_stability_paths(self):
        plt.plot(self._alphas, self._stability_path_matrix.T);

    def get_features_from_dataframe(self, dataframe):

        coefficients_lasso = pd.DataFrame(
            self._stability_path_matrix.T,
            index=np.round(self._alphas, 5),
            columns=dataframe.columns
        )

        return coefficients_lasso.columns[self.feature_mask]