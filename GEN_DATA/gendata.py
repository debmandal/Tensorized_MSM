import numpy as np
import tensorly as tl
import scipy as sp
import math as math
import random
import datetime
from scipy.special import logit, expit
import argparse



def gen_tensor(N,T,B,r,L):
# construct tensor of rank r and of size (N,T,B) with singular values bounded between [1,L]
    UN = np.random.uniform(size=(N,r))
    UT = np.random.uniform(size=(T,r))
    UB = np.random.uniform(size=(B,r))

    #normalize
    UN_normed = UN/np.linalg.norm(UN,axis = 0)
    UT_normed = UT/np.linalg.norm(UT,axis = 0)
    UB_normed = UB/np.linalg.norm(UB, axis = 0)

    wts_TT = np.random.uniform(low=L/4,high=L,size=r)

    TT = tl.kruskal_to_tensor((wts_TT, [UN_normed, UT_normed, UB_normed] ) )

    return(TT)

def gen_data(TT, policy):
    N,T,B = TT.shape
    #generate covariates, outcomes, and treatments

    d = 4 #dimension of the covariate
    k = np.int(np.log2(B)) #length of the history

    XX = np.ndarray(shape=(N,T,d))
    Aobs = np.zeros((N,T), dtype='int')
    Yobs = np.zeros((N,T), dtype='float')
    gamma = [1, -0.5, 0.25, 0.1]
    delta = [27.4, 13.7, 13.7, 13.7]

    #assign values for Yobs, Aobs, and XX
    #we assume Aobs_{i,-1} = 0 for all i
    if policy == 'I':
        for i in range(N):
            for t in range(T):
                #generate XX[i,t,]
                z = np.random.normal(0,1,4)
                if t  == 0:
                    u = 1 #there was a mistake in the first run
                else:
                    u =  2 + (2 * Aobs[i][t-1] - 1)/3.0

                XX[i,t,0] = z[0] * u
                XX[i,t,1] = z[1] * u
                XX[i,t,2] = np.abs(z[2] * u)
                XX[i,t,3] = np.abs(z[3] * u)

                #generate Aobs[i][t]
                if t == 0:
                    prob = expit(np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
                else:
                    prob = expit(-Aobs[i,t-1] + np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
                Aobs[i][t] = np.random.binomial(1, prob)

                #generate Yobs[i][t]
                history = np.zeros(k, dtype='int')

                for idx in range(np.min([t+1,k])):
                    history[k-idx-1] = Aobs[i][t-idx]
                hist_conv = np.int(''.join(str(e) for e in history), 2)
                Yobs[i][t] = TT[i, t, hist_conv] #+ np.dot(XX[i,t,], beta) + np.random.normal(loc=0,scale=1,size=1)

                sum_treat = 0
                for idx in range(np.min([3,t])):
                    sum_treat += Aobs[i][t-idx]

                sum_cov = 0
                for idx in range(np.min([3,t])):
                    sum_cov += np.dot(delta, XX[i,t-idx,])

                Yobs[i][t] += 250 - 10 * sum_treat + sum_cov + np.random.normal(loc=0,scale=0,size=1)

    elif policy == 'II':
        for i in range(N):
            for t in range(T):
                #generate XX[i,t,]
                z = np.random.normal(0,1,4)
                if t == 0:
                    u = 1 #there was a mistake in the first run
                elif t == 1:
                    u = 2 + (2 * Aobs[i][t-1] - 1)/3.0
                else:
                    u = 2 + (2 * Aobs[i][t-1] - 1)/3.0
                    u = u * (2 + (2 * Aobs[i][t-2] - 1)/3.0)

                XX[i,t,0] = z[0] * u
                XX[i,t,1] = z[1] * u
                XX[i,t,2] = np.abs(z[2] * u)
                XX[i,t,3] = np.abs(z[3] * u)

                #generate Aobs[i][t]
                if t == 0:
                    unnorm_prob = np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1)
                elif t == 1:
                    unnorm_prob = np.dot(gamma, XX[i,t,]) + Aobs[i,t-1]
                    unnorm_prob += (np.dot(gamma, XX[i,t-1,]) ) / 2
                    unnorm_prob += (-0.5)**(t+1)
                elif t == 2:
                    unnorm_prob = np.dot(gamma, XX[i,t,]) + Aobs[i,t-1]
                    unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + Aobs[i,t-2]) / 2
                    unnorm_prob += (np.dot(gamma, XX[i,t-2,]) ) / 4
                    unnorm_prob += (-0.5)**(t+1)
                else:
                    unnorm_prob = np.dot(gamma, XX[i,t,]) + Aobs[i,t-1]
                    unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + Aobs[i,t-2]) / 2
                    unnorm_prob += (np.dot(gamma, XX[i,t-2,]) + Aobs[i,t-3]) / 4
                    unnorm_prob += (-0.5)**(t+1)
                prob = expit(unnorm_prob)
                Aobs[i][t] = np.random.binomial(1, prob)

                #generate Yobs[i][t]
                history = np.zeros(k, dtype='int')

                for idx in range(np.min([t+1,k])):
                    history[k-idx-1] = Aobs[i][t-idx]
                hist_conv = np.int(''.join(str(e) for e in history), 2)
                Yobs[i][t] = TT[i, t, hist_conv] #+ np.dot(XX[i,t,], beta) + np.random.normal(loc=0,scale=1,size=1)

                sum_treat = 0
                for idx in range(np.min([3,t])):
                    sum_treat += Aobs[i][t-idx]

                sum_cov = 0
                for idx in range(np.min([3,t])):
                    sum_cov += np.dot(delta, XX[i,t-idx,])

                Yobs[i][t] += 250 - 10 * sum_treat + sum_cov + np.random.normal(loc=0,scale=0,size=1)

    return Aobs, Yobs, XX


def get_atet(TT, A, niter, policy):
    N,T,B = TT.shape

    d = 4 #dimension of the covariate
    k = np.int(np.log2(B)) #length of the history

    gamma = [1, -0.5, 0.25, 0.1]
    delta = [27.4, 13.7, 13.7, 13.7]

    effects = np.zeros(niter)
    #assign values for Yobs, Aobs, and XX
    #we assume Aobs_{i,-1} = 0 for all i

    #count number of ones
    num_of_ones = 0
    for i in range(N):
        for t in range(T):
            if A[i][t] == 1:
                num_of_ones += 1

    for iiter in range(niter):
        XX = np.ndarray(shape=(N,T,d) )
        Dobs = np.zeros((N,T), dtype='int')
        for i in range(N):
            for t in range(T):
                if A[i][t] == 1:#estimate E[Y_{it}(A[i,t-k+1:t])] - E[Y_{it}(A[i,t-k+1:t-1,0])]
                    if policy == 'I':
                        for tp in range(t):

                            #generate XX[i,tp,]
                            z = np.random.normal(0,1,4)
                            if tp  == 0:
                                u = 1
                            else:
                                u = 2 + (2 * A[i][tp-1] - 1)/3.0

                            XX[i,tp,0] = z[0] * u
                            XX[i,tp,1] = z[1] * u
                            XX[i,tp,2] = np.abs(z[2] * u)
                            XX[i,tp,3] = np.abs(z[3] * u)
                            #end tp for 0 to t-1
                        #generate XX[i,t,] when A[i][t] equals 1
                        z = np.random.normal(0,1,4)
                        if t == 0:
                            u = 1
                        else:
                            u = 2 + 1/3.0

                        XX[i,t,0] = z[0] * u
                        XX[i,t,1] = z[1] * u
                        XX[i,t,2] = np.abs(z[2] * u)
                        XX[i,t,3] = np.abs(z[3] * u)
                    elif policy == 'II':
                        for tp in range(t):
                            #generate XX[i,tp,]
                            z = np.random.normal(0,1,4)
                            if tp == 0:
                                u = 1 #there was a mistake in the first run
                            elif tp == 1:
                                u = 2 + (2 * A[i][tp-1] - 1)/3.0
                            else:
                                u = 2 + (2 * A[i][tp-1] - 1)/3.0
                                u = u * (2 + (2 * A[i][tp-2] - 1)/3.0)

                            XX[i,tp,0] = z[0] * u
                            XX[i,tp,1] = z[1] * u
                            XX[i,tp,2] = np.abs(z[2] * u)
                            XX[i,tp,3] = np.abs(z[3] * u)

                        #generate XX[i,t,] when A[i][t] equals 1
                        z = np.random.normal(0,1,4)
                        if t == 0:
                            u = 1 #there was a mistake in the first run
                        elif t == 1:
                            u = 2 + 1/3.0
                        else:
                            u = 2 + 1/3.0
                            u = u * (2 + (2 * A[i][t-2] - 1)/3.0)

                        XX[i,t,0] = z[0] * u
                        XX[i,t,1] = z[1] * u
                        XX[i,t,2] = np.abs(z[2] * u)
                        XX[i,t,3] = np.abs(z[3] * u)



                    #generate the counterfactual outcomes, this step is the same for both the policies
                    #generate Yobs[i][t] when A[i][t] is set to 1
                    history = np.zeros(k, dtype='int')
                    history[k-1] = 1

                    for idx in range(1, np.min([t,k])):
                        history[k-idx-1] = A[i][t-idx]
                    hist_conv = np.int(''.join(str(e) for e in history), 2)
                    Yobs_it_1 = TT[i, t, hist_conv] #+ np.dot(XX[i,t,], beta) + np.random.normal(loc=0,scale=1,size=1)

                    sum_treat = 1
                    for idx in range(1,np.min([3,t])):
                        sum_treat += A[i][t-idx]

                    sum_cov = 0
                    for idx in range(np.min([3,t])):
                        sum_cov += np.dot(delta, XX[i,t-idx,])

                    Yobs_it_1 += 250 - 10 * sum_treat + sum_cov + np.random.normal(loc=0,scale=0,size=1)

                    #generate Yobs[i][t] when A[i][t] is set to 0
                    #we first need to generate the covariate XX[i,t,] which is different for the two policies
                    if policy == 'I':
                        z = np.random.normal(0,1,4)
                        if t == 0:
                            u = 0
                        else:
                            u = 2 - 1/3.0

                        XX[i,t,0] = z[0] * u
                        XX[i,t,1] = z[1] * u
                        XX[i,t,2] = np.abs(z[2] * u)
                        XX[i,t,3] = np.abs(z[3] * u)
                    elif policy == 'II':
                        z = np.random.normal(0,1,4)
                        if t == 0:
                            u = 1 #there was a mistake in the first run
                        elif t == 1:
                            u = 2 - 1/3.0
                        else:
                            u = 2 - 1/3.0
                            u = u * (2 + (2 * A[i][t-2] - 1)/3.0)

                        XX[i,t,0] = z[0] * u
                        XX[i,t,1] = z[1] * u
                        XX[i,t,2] = np.abs(z[2] * u)
                        XX[i,t,3] = np.abs(z[3] * u)

                    #now generate Yobs[i][t] when A[i][t] is set to 0, this step is the same for both the policies

                    history = np.zeros(k, dtype='int')

                    for idx in range(1, np.min([t,k])):
                        history[k-idx-1] = A[i][t-idx]
                    hist_conv = np.int(''.join(str(e) for e in history), 2)
                    Yobs_it_0 = TT[i, t, hist_conv] #+ np.dot(XX[i,t,], beta) + np.random.normal(loc=0,scale=1,size=1)

                    sum_treat = 0
                    for idx in range(1,np.min([3,t])):
                        sum_treat += A[i][t-idx]

                    sum_cov = 0
                    for idx in range(np.min([3,t])):
                        sum_cov += np.dot(delta, XX[i,t-idx,])

                    Yobs_it_0 += 250 - 10 * sum_treat + sum_cov + np.random.normal(loc=0,scale=0,size=1)

                    #update Dobs
                    Dobs[i][t] = Yobs_it_1 - Yobs_it_0
#                    Dobs[i][t] = (Dobs[i][t] * iiter + (Yobs_it_1 - Yobs_it_0) ) / (1 + iiter)

        #compute atet
        effects[iiter] = np.sum(Dobs) / num_of_ones

    atet = np.mean(effects)
    serr = np.std(effects) / np.sqrt(niter)
    #print(str(atet) + ' +- ' + str(serr))

    return atet, serr




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--it", help="index of the instance (0 to 99)")
    parser.add_argument("--world", help="type of world (fat or thin)")
    parser.add_argument("--policy", help="type of policy (I or II)")

    args = parser.parse_args()
    if(args.it):
        it = int(args.it)
    else:
        print('Need it')
        exit()

    if(args.world):
        world = str(args.world)
    else:
        print('Need world!')
        exit()

    if(args.policy):
        policy = str(args.policy)
    else:
        print('Need policy')
        exit()


    #set up parameters
    if world == 'thin':
        N = 500
        T = 10
    elif world == 'fat':
        N = 10
        T = 500

    k = 5  #length of relevant history
    B = 2**k #third dimension
    r = 10  #rank of the underlying tensor
    L = 200 #maximum absolute value of a tensor


    if world == 'thin':
        TT = np.load('./data/thin/TT_thin.npy')
    elif world == 'fat':
        TT = np.load('./data/fat/TT_fat.npy')

    Aobs, Yobs, XX = gen_data(TT, policy)
    #save data
    if world == 'thin':
        np.save('./data/thin/Aobs_thin_' + policy + '_' + str(it) + '.npy', Aobs)
        np.save('./data/thin/Yobs_thin_' + policy + '_' + str(it) + '.npy', Yobs)
        np.save('./data/thin/XX_thin_'   + policy + '_' + str(it) + '.npy', XX)
    elif world == 'fat':
        np.save('./data/fat/Aobs_fat_' + policy + '_' + str(it) + '.npy', Aobs)
        np.save('./data/fat/Yobs_fat_' + policy + '_' + str(it) + '.npy', Yobs)
        np.save('./data/fat/XX_fat_' + policy + '_' + str(it) + '.npy', XX)


    atet_cur, serr = get_atet(TT, Aobs, 1000, policy)

    print(str(atet_cur) + '+-' + str(serr))

    if world == 'thin':
        np.save('./data/thin/atet_thin_' + policy + '_' + str(it) + '.npy', atet_cur)
    elif world == 'fat':
        np.save('./data/fat/atet_fat_' + policy + '_' + str(it) + '.npy', atet_cur)
