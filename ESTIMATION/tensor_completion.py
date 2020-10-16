import numpy as np
import tensorly as tl
import scipy as sp
import math as math
import random
import datetime
import argparse
from scipy.special import logit, expit
from tensorly.decomposition import parafac
from progressbar import printProgressBar
from wts_estimation import get_weights


def weighted_tensor_completion(A, Y, wts, WW, L, r, N_ITER=500):
    N,T,B = WW.shape
    k = np.int(np.log2(B)) #length of the history
    #initialization
    U_tensor_n = np.random.uniform(size=(N,r))
    U_tensor_t = np.random.uniform(size=(T,r))
    U_tensor_b = np.random.uniform(size=(B,r))

    U_tensor_n_normed = U_tensor_n / np.linalg.norm(U_tensor_n, axis=0)
    U_tensor_t_normed = U_tensor_t / np.linalg.norm(U_tensor_t, axis=0)
    U_tensor_b_normed = U_tensor_b / np.linalg.norm(U_tensor_b, axis=0)

    U_wts = np.random.uniform(low=L/4, high=L, size=r)

    tensor = tl.kruskal_to_tensor((U_wts, [U_tensor_n_normed, U_tensor_t_normed, U_tensor_b_normed] ) )

    #prepare yw tensor
    yw_tensor = tl.zeros_like(tensor)

    for i in range(N):
        for t in range(T):
            history = np.zeros(k)
            low = np.min([k,t+1])
            for j in range(low):
                history[k-j-1] = A[i][t-j]
            str1 = ''.join(str(int(e)) for e in history)
            p = np.int(str1, 2)
            yw_tensor[i,t,p] = (wts[i][t] * Y[i][t]) / WW[i,t,p]**2

    step_size = 0.02
    it_tensor = tensor
    prev_loss = tl.norm(WW * (yw_tensor - it_tensor))


    items = list(range(N_ITER))
    l_items = len(items)

    for it in range(N_ITER):
        #compute grad
        grad_tensor = -2 * (WW**2) * (yw_tensor - it_tensor)
        it_tensor -= step_size * grad_tensor
        #project into rank r space
        lambdas, factors = parafac(it_tensor, rank=r, init='svd', tol=10e-8, normalize_factors=True)
        #clip the weights
        lambdas = np.clip(lambdas, a_min = L/6, a_max=1.5*L)
        it_tensor = tl.kruskal_to_tensor((lambdas, factors ))

        new_loss = tl.norm(WW * (yw_tensor - it_tensor))
        if it >= 20:
            if np.abs(prev_loss - new_loss) / prev_loss <= 10e-6:
                break
            elif new_loss > prev_loss * 1.005:
                break

        prev_loss = new_loss
        #if it % 25 == 0:
        #   printProgressBar(it, l_items, prefix = 'Progress:', suffix = 'Complete', length = 50)

    #printProgressBar(it, l_items, prefix = 'Progress:', suffix = 'Complete', length = 50)
    #print()

    return it_tensor, new_loss

def estimated_atet(A,tensor):
    N,T,B = tensor.shape
    k = np.int(np.log2(B)) #length of the history

    effect = 0.0
    count = 0
    for i in range(N):
        for t in range(T):
            if A[i][t] == 1:
                history = np.zeros(k, dtype='int')
                for idx in range(np.min([t,k])):
                    history[k-idx-1] = A[i][t-idx]
                hist_conv_1 = np.int(''.join(str(e) for e in history), 2)
                history[k-1] = 0
                hist_conv_0 = np.int(''.join(str(e) for e in history), 2)
                effect += (tensor[i,t,hist_conv_1] - tensor[i,t,hist_conv_0])
                count += 1

    return effect / count

def run_outiter(WW, it, L, world, policy):
    #run for all possible values of assumed rank r
    est_atet = np.zeros(10)
    k = 5

    if world == 'thin':
        Aobs = np.load('../GEN_DATA/data/thin/Aobs_thin_' + policy + '_' + str(it) + '.npy')
        Yobs = np.load('../GEN_DATA/data/thin/Yobs_thin_' + policy + '_' +  str(it) + '.npy')
        XX = np.load('../GEN_DATA/data/thin/XX_thin_' + policy + '_' + str(it) + '.npy')
        # get the weights
        wts = get_weights(Aobs, XX, k, policy)
    elif world == 'fat':
        Aobs = np.load('../GEN_DATA/data/fat/Aobs_fat_' + policy + '_' + str(it) + '.npy')
        Yobs = np.load('../GEN_DATA/data/fat/Yobs_fat_' + policy + '_' + str(it) + '.npy')
        XX = np.load('../GEN_DATA/data/fat/XX_fat_' + policy + '_' + str(it) + '.npy')
        # get the weights
        wts = get_weights(Aobs, XX, k, policy)

    for r in range(5,15):

        TT, loss = weighted_tensor_completion(Aobs, Yobs, wts, WW, L, r, N_ITER=500)

        for j in range(1,10):
            new_TT, new_loss = weighted_tensor_completion(Aobs, Yobs, wts, WW, L, r, N_ITER=500)

            if new_loss < loss:
                TT = new_TT
                loss = new_loss
        est_atet[r-5] = estimated_atet(Aobs,TT)
        #print()
        print('Estimated ATET for iteration ' + str(it) + ', rank ' + str(r) +  ', policy ' + policy + ', world ' + world + ': ' + str(est_atet[r-5]))

    if world == 'thin':
        np.save('models/thin/result_est_atet_thin_' + policy + '_' + str(it) + '.npy', est_atet)
    elif world == 'fat':
        np.save('models/fat/result_est_atet_fat_' + policy + '_' + str(it) + '.npy', est_atet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--it", help="index of the instance (0 to 99)")
    parser.add_argument("--world", help="type of world (fat or thin)")
    parser.add_argument("--policy", help="policy type (I or II)")


    L = 200
    k = 5

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

    if world == 'thin':
        WW = np.load('./models/thin/WW_thin_' + policy + '_' + str(k) + '.npy')
    else:
        WW = np.load('./models/fat/WW_fat_' + policy + '_' + str(k) + '.npy')


    lbd = 10e-9
    WW = (1-lbd) * WW + lbd * tl.ones(WW.shape)


    run_outiter(WW, it, L, world, policy)


