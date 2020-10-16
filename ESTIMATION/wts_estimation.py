import numpy as np
import tensorly as tl
import scipy as sp
import math as math
import random
import datetime
from scipy.special import logit, expit
from progressbar import printProgressBar
import argparse


def get_prob(A, XX, i, t, policy):

    gamma = [1, -0.5, 0.25, 0.1]
    delta = [27.4, 13.7, 13.7, 13.7]
    if policy == 'I':
        if t == 0:
            prob = expit(np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
        else:
            prob = expit(-A[i,t-1] + np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
    elif policy == 'II':
        if t == 0:
            unnorm_prob = np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1)
        elif t == 1:
            unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
            unnorm_prob += (np.dot(gamma, XX[i,t-1,]) ) / 2
            unnorm_prob += (-0.5)**(t+1)
        elif t == 2:
            unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
            unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + A[i,t-2]) / 2
            unnorm_prob += (np.dot(gamma, XX[i,t-2,]) ) / 4
            unnorm_prob += (-0.5)**(t+1)
        else:
            unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
            unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + A[i,t-2]) / 2
            unnorm_prob += (np.dot(gamma, XX[i,t-2,]) + A[i,t-3]) / 4
            unnorm_prob += (-0.5)**(t+1)
        prob = expit(unnorm_prob)


    if A[i][t] == 1:
        return prob
    else:
        return 1-prob

def get_weights(A,XX,k, policy):
    gamma = [1, -0.5, 0.25, 0.1]
    delta = [27.4, 13.7, 13.7, 13.7]
    N, T = A.shape

    wts = np.ones((N,T), dtype='float')
    niters = 500
    dim = XX.shape
    d = dim[2]

    for i in range(N):
        for t in range(T):
            numer = 1.0
            denom = 1.0
            for p in range(min([t+1,k])):
                #estimate denominator for time t-p
                denom = get_prob(A, XX, i, t-p, policy)

                if policy == 'I':
                    sum_numer = 0.0
                    for it in range(niters):
                    #estimate numerator for time t-p
                        x = np.zeros(d, dtype='float')
                        z = np.random.normal(0,1,4)
                        if (t-p) == 0:
                            u = 1
                        else:
                            u = 2 + (2 * A[i][t-p-1] - 1)/3.0

                        x[0] = z[0] * u
                        x[1] = z[1] * u
                        x[2] = np.abs(z[2] * u)
                        x[3] = np.abs(z[3] * u)

                        if t-p == 0:
                            prob = expit(np.dot(gamma, x) + (-0.5)**(t-p+1) )
                        else:
                            prob = expit(-A[i,t-p-1] + np.dot(gamma, x) + (-0.5)**(t-p+1) )
                        sum_numer += prob

                    if A[i][t] == 1:
                        numer = sum_numer / niters
                    else:
                        numer = 1 - (sum_numer / niters)
                elif policy == 'II':
                    sum_numer = 0
                    for it in range(niters):
                        #estimate numerator for time t-p
                        x = np.zeros(d, dtype='float')
                        z = np.random.normal(0,1,4)
                        if (t-p) == 0:
                            u = 1
                        elif (t-p) == 1:
                            u = 2 + (2 * A[i][t-p-1] - 1)/3.0
                        else:
                            u = 2 + (2 * A[i][t-p-1] - 1)/3.0
                            u = u * (2 + (2 * A[i][t-p-2] - 1)/3.0)

                        x[0] = z[0] * u
                        x[1] = z[1] * u
                        x[2] = np.abs(z[2] * u)
                        x[3] = np.abs(z[3] * u)

                        if (t-p) == 0:
                            unnorm_prob = np.dot(gamma, x) + (-0.5)**(t-p+1)
                        elif (t-p) == 1:
                            unnorm_prob = np.dot(gamma, x) + A[i,t-p-1]
                            unnorm_prob += (np.dot(gamma, x) ) / 2
                            unnorm_prob += (-0.5)**(t-p+1)
                        elif (t-p) == 2:
                            unnorm_prob = np.dot(gamma, x) + A[i,t-p-1]
                            unnorm_prob += (np.dot(gamma, x)  + A[i,t-p-2]) / 2
                            unnorm_prob += (np.dot(gamma, x) ) / 4
                            unnorm_prob += (-0.5)**(t-p+1)
                        else:
                            unnorm_prob = np.dot(gamma, x) + A[i,t-p-1]
                            unnorm_prob += (np.dot(gamma, x)  + A[i,t-p-2]) / 2
                            unnorm_prob += (np.dot(gamma, x) + A[i,t-p-3]) / 4
                            unnorm_prob += (-0.5)**(t-p+1)
                        prob = expit(unnorm_prob)
                        sum_numer += prob

                    if A[i][t] == 1:
                        numer = sum_numer / niters
                    else:
                        numer = 1 - (sum_numer / niters)

                wts[i][t] *= (numer / denom)

    #might want to take convex combination with low prob

    return wts


def get_WW(N,T,k,policy,num_times=1000):
    gamma = [1, -0.5, 0.25, 0.1]
    delta = [27.4, 13.7, 13.7, 13.7]

    B = 2**k
    WW = tl.zeros((N,T,B))
    d = 4

    counts = tl.zeros_like(WW)


    items = list(range(num_times))
    l_items = len(items)
    for it in range(num_times):
        if it % 50 == 0:
            printProgressBar(it, l_items, prefix='Progress:', suffix='Complete', length=50)
        #print(iter)
        A = np.zeros((N,T),dtype=int)
        XX = np.ndarray(shape=(N,T,d))
        #Assign values for A, Y, XX
        #We assume A[i,-1] = 0 for all i
        if policy == 'I':
            for i in range(N):
                for t in range(T):
                #generate XX[i,t,]
                    z = np.random.normal(0,1,4)
                    if t  == 0:
                        u = 1 #there was a mistake in the first run
                    else:
                        u =  2 + (2 * A[i][t-1] - 1)/3.0

                    XX[i,t,0] = z[0] * u
                    XX[i,t,1] = z[1] * u
                    XX[i,t,2] = np.abs(z[2] * u)
                    XX[i,t,3] = np.abs(z[3] * u)

                    #generate A[i][t]
                    if t == 0:
                        prob = expit(np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
                    else:
                        prob = expit(-A[i,t-1] + np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1))
                    A[i][t] = np.random.binomial(1, prob)
        elif policy == 'II':
            for i in range(N):
                for t in range(T):
                    #generate XX[i,t,]
                    z = np.random.normal(0,1,4)
                    if t == 0:
                        u = 1 #there was a mistake in the first run
                    elif t == 1:
                        u = 2 + (2 * A[i][t-1] - 1)/3.0
                    else:
                        u = 2 + (2 * A[i][t-1] - 1)/3.0
                        u = u * (2 + (2 * A[i][t-2] - 1)/3.0)

                    XX[i,t,0] = z[0] * u
                    XX[i,t,1] = z[1] * u
                    XX[i,t,2] = np.abs(z[2] * u)
                    XX[i,t,3] = np.abs(z[3] * u)

                    #generate A[i][t]
                    if t == 0:
                        unnorm_prob = np.dot(gamma, XX[i,t,]) + (-0.5)**(t+1)
                    elif t == 1:
                        unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
                        unnorm_prob += (np.dot(gamma, XX[i,t-1,]) ) / 2
                        unnorm_prob += (-0.5)**(t+1)
                    elif t == 2:
                        unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
                        unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + A[i,t-2]) / 2
                        unnorm_prob += (np.dot(gamma, XX[i,t-2,]) ) / 4
                        unnorm_prob += (-0.5)**(t+1)
                    else:
                        unnorm_prob = np.dot(gamma, XX[i,t,]) + A[i,t-1]
                        unnorm_prob += (np.dot(gamma, XX[i,t-1,])  + A[i,t-2]) / 2
                        unnorm_prob += (np.dot(gamma, XX[i,t-2,]) + A[i,t-3]) / 4
                        unnorm_prob += (-0.5)**(t+1)
                    prob = expit(unnorm_prob)
                    A[i][t] = np.random.binomial(1, prob)

        for i in range(N):
            for t in range(T):
                history = np.zeros(k, dtype='int')
                for idx in range(np.min([k,t+1])):
                    history[k-idx-1] = A[i][t-idx]
                str1 = ''.join(str(int(e)) for e in history)
                p = np.int(str1, 2)
                counts[i,t,p] = counts[i,t,p] + 1


    printProgressBar(num_times, l_items, prefix='Progress:', suffix='Complete', length=50)
    print()
    WW = counts/num_times
    return WW**(0.5)


'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", help="type of world (fat or thin)")
    parser.add_argument("--policy", help="type of policy (I or II)")

    args = parser.parse_args()

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


    #set up parameters
    k = 5  #length of relevant history
    B = 2**k #third dimension
    r = 10  #rank of the underlying tensor
    L = 200 #maximum absolute value of a tensor


    for pk in range(2,9):
        WW = get_WW(N,T,k,policy)
        if world == 'thin':
            np.save('models/thin/WW_thin_' + policy + '_' + str(pk) + '.npy', WW)
        elif world == 'fat':
            np.save('models/fat/WW_fat_' + policy + '_' + str(pk) + '.npy', WW)


'''
