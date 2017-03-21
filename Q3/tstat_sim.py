import numpy as np
import matplotlib.pyplot as plt

from tstat_functions import TStat, AUC, load_data

# Sim Data
X = load_data()
Y = np.ones(40)
Y[20:] += 1

step = 0.05
AUCs = []
s0s = np.arange(0, 1 + step, step)

for s0 in s0s:
	T = TStat(X, Y, s0)
	NP = np.zeros(X.shape[0])
	NP[0:500] = 1
	AUCs.append(AUC(T, NP))

plt.plot(s0s, AUCs)
plt.plot(s0s, AUCs,'+', color='red')
plt.title(r'AUC relative to $s_0$')
plt.xlabel(r'$s_0$')
plt.ylabel(r'AUC')

plt.show()

print(AUCs)