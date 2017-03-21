import numpy as np
import matplotlib.pyplot as plt

from tstat_functions import TStat, AUC, load_data

def norm_prob(T, mean ,var):
	pdf = np.exp(-(T-mean)**2/(2*var))/np.sqrt(2*var*np.pi)
	return pdf

# Sim Data
X = load_data(filename='cancerData.tsv')
Y = load_data(filename='cancerDataGrp.tsv')

rows = X.shape[0]
cols = X.shape[1]

max_s0 = 0.3
steps = 19
step_size = max_s0 / steps
s0s = np.arange(0, max_s0 + step_size, step_size)

n_perm = 20

sig_num = []
for s0 in s0s:
	t_stats = np.empty([rows, n_perm + 1])
	t_stats[:,0] = abs(TStat(X,Y,s0))

	num_sigs = 0

	np.random.seed(42069)
	for i in range(1,t_stats.shape[1]):
		Yi = np.random.permutation(Y)
		t_stats[:,i] = abs(TStat(X,Yi,s0))

	for i in range(0, rows):
		if len(np.where(t_stats[i,1:] >= t_stats[i,0])[0])/n_perm < 0.1:
			num_sigs += 1
	sig_num.append(num_sigs)
print(max(sig_num))

plt.plot(np.arange(0, max_s0 + step_size, step_size), sig_num)
plt.title('FDR < 0.1 Genes Relative to $s_0$')
plt.xlabel(r'$s_0$')
plt.ylabel('Number of Genes with FDR < 0.1')
plt.show()