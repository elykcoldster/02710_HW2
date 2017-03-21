import numpy as np

def s(x1, x2):
	n1 = x1.shape[1]
	n2 = x1.shape[1]
	s1 = np.std(x1, axis=1)
	s2 = np.std(x2, axis=1)
	s = np.sqrt(((n1-1)*s1**2+(n2-1)*s2**2)/(n1+n2-2))
	return s

def TStat(X, Y, s0):
	X1 = X[:,Y == 1]
	X2 = X[:,Y == 2]
	n1 = X1.shape[1]
	n2 = X2.shape[1]
	Xbar1 = np.mean(X1, axis=1)
	Xbar2 = np.mean(X2, axis=1)
	t = (Xbar1-Xbar2)/(s0 + s(X1,X2)*np.sqrt(1/n1+1/n2))
	return t

def load_data(filename='simData.tsv'):
    X = np.loadtxt(filename)
    return (X)

def AUC(T, NP):
	Np = sum(NP == 1)
	Nn = sum(NP == 0)

	T = abs(T)
	ranks = np.argsort(T)
	p_rank_sum = 0
	for i in range(0, len(NP)):
		rank = ranks[i]
		if NP[rank] == 1:
			p_rank_sum += i
	return p_rank_sum/(Np * Nn)