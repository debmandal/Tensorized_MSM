import numpy as np
import scipy as sp
import math as math

def printArray(arr):
	n = len(arr)

	print('(' + str(arr[0]), end='')
	for i in range(1,n):
		print(', ' + str(arr[i]), end='')
	print(')')

if __name__ == '__main__':

	worlds = ['thin', 'fat']
	policies = ['I', 'II']

	for w in worlds:
		for pol in policies:
			est_atets = np.zeros((100,10))

			diff_atet = np.zeros((100,10))
			for it in range(100):
				est_atets[it] = np.load('../ESTIMATION/models/' + w + '/result_est_atet_' + w + '_' + pol + '_' + str(it) + '.npy')
				#print(est_atets[it])
				atet_cur = np.load('../GEN_DATA/data/' + w + '/atet_' + w + '_' + pol + '_' + str(it) + '.npy')
				#print(atet_cur)
				diff_atet[it] = (atet_cur - est_atets[it])/atet_cur

			print(w + '' + pol + 'mean <- c', end='')
			printArray(np.mean(np.square(diff_atet), axis=0) )
			print(w + '' + pol + 'se <- c', end='')
			printArray(np.std(np.square(diff_atet), axis=0)/10)
			print(' ')
			#print(np.mean(np.abs(diff_atet), axis=0))
			#print(np.std(np.abs(diff_atet), axis=0) / 10)

	print('History:')
	for w in worlds:
		for pol in policies:
			est_atets = np.zeros((100,7))

			diff_atet = np.zeros((100,7))
			for it in range(100):
				est_atets[it] = np.load('../ESTIMATION/models/' + w + '/result_hist_est_atet_' + w + '_' + pol + '_' + str(it) + '.npy')
				#print(est_atets[it])
				atet_cur = np.load('../GEN_DATA/data/' + w + '/atet_' + w + '_' + pol + '_' + str(it) + '.npy')
				#print(atet_cur)
				diff_atet[it] = (atet_cur - est_atets[it])/atet_cur
				#diff_atet[it] = (atet_cur - est_atets[it])

			print(w + '' + pol + 'mean <- c', end='')
			printArray(np.mean(np.square(diff_atet), axis=0))
			print(w + '' + pol + 'se <- c', end='')
			printArray(np.std(np.square(diff_atet), axis=0)/10)
			print(' ')
			#print(np.mean(np.abs(diff_atet), axis=0))
			#print(np.std(np.abs(diff_atet), axis=0) / 10)



