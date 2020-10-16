import numpy as np
import scipy as sp
import math as math
import random
from sklearn import linear_model
import statsmodels.api as sm




get_indices = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
#Weighted Tensor Completion Through Gradient Descent
if __name__ == '__main__':

    worlds = ['thin', 'fat']
    policies = ['I', 'II']

    for w in worlds:
        if w == 'thin':
            T = 10
        else:
            T = 500
        for pol in policies:
            result = np.zeros(100)
            print(w + '_' + pol)
            for it in range(1,100):
                #print(w + pol + str(it))
                Aobs = np.load('../GEN_DATA/data/' + w + '/Aobs_' + w + '_' + pol + '_' + str(it) + '.npy')
                Yobs = np.load('../GEN_DATA/data/' + w + '/Yobs_' + w + '_' + pol + '_' + str(it) + '.npy')
                atet_cur = np.load('../GEN_DATA/data/' + w + '/atet_' + w + '_' + pol + '_' + str(it) + '.npy')
                wts = np.load('../ESTIMATION/models/' + w + '/wts_' + w + '_' + pol + '_' + str(it) + '.npy')
                #print(atet_cur)

                effect = 0.0
                for t in range(T): #fit beta_t
                    #print(Yobs[:,t])
                    reg = linear_model.LinearRegression(fit_intercept=True, normalize=True)
                    reg.fit(Aobs[:,0:t+1],Yobs[:,t], sample_weight=wts[:,t])
                    #glm = sm.GLM(Yobs[:,t], Aobs[:,0:t+1],  family=sm.families.InverseGaussian())
                    #res = glm.fit()
                    #get prediction for the treated
                    if np.count_nonzero(Aobs[:,t]) > 0:
                        indices = get_indices(1,Aobs[:,t])
                        treated = Aobs[indices,0:t+1]
                        controlled = np.hstack((Aobs[indices,0:t], np.zeros((len(indices), 1), dtype=int )) )
                        effect += np.sum(reg.predict(treated) - reg.predict(controlled) )

                result[it] = ((effect / np.sum(Aobs)) - atet_cur)/atet_cur
            print(np.mean(np.square(result)) )
            print(1.95*np.std(np.square(result )) /10)






