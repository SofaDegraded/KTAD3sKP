import numpy as np
from scipy.stats.stats import ks_2samp
import math as mth
from scipy import stats as scst
from statsmodels.distributions.empirical_distribution import ECDF

def reading_init_param(fileName):
    data = np.loadtxt(fileName,skiprows=2)
    size = data.size
    return size, data

def round_new(data, decimals):
    return np.array([round(x, decimals) for x in data])
#моделирование выборок с заданной точностью
def modeling(mode, N, xi = 0, eta = 1): 
    
    if mode == 0:
        #data = round_new(scst.norm.rvs(loc=xi, scale=eta, size=N), 6)
        data = scst.norm.rvs(loc=xi, scale=eta, size=N)
    elif mode == 1: 
        #data = round_new(scst.expon.rvs(loc=xi, scale=eta, size=N), 6)
        data = scst.expon.rvs(loc=xi, scale=eta, size=N)
    elif mode == 2:
        #data = round_new(scst.cauchy.rvs(loc=xi, scale=eta, size=N), 6)
        data = scst.cauchy.rvs(loc=xi, scale=eta, size=N)
    return data

#вычисление статистики критерия
def t_student_calc_s_star(data1, data2):
    n1 = len(data1) # размер первой выборки
    n2 = len(data2) # размер второй выборки
    x1_mean = data1.mean()
    x2_mean = data2.mean()
    s1 = data1.var()
    s2 = data2.var()
    s_star = (x1_mean - x2_mean) / (mth.sqrt(s1 / n1 + s2 / n2)) #статистика критерия
    if n1 != n2:
        v = (s1 / n1 + s2 / n2)**2 / ((s1 / n1)**2 / (n1-1)+ (s2 / n2)**2/(n2-1))
    else:
        v = n1-1+(2*n1-2)/(s1/s2+s2/s1)
    return s_star, v

def Wilcoxon_calc_s_star(data1, data2):
    n1 = len(data1) # размер первой выборки
    n2 = len(data2) # размер второй выборки
    data = np.sort(np.concatenate((data1, data2))) # объединенная выборка
    r1 = np.sum(list(map(lambda x: x+1, np.where(np.isin(data, data1))[0])))
    r2 = np.sum(list(map(lambda x: x+1, np.where(np.isin(data, data2))[0])))
    u1 = n1*n2 + n1*(n1+1)/2 - r1
    u2 = n1*n2 + n2*(n2+1)/2 - r2
    s_star =  min(u1, u2) #статистика критерия
    return s_star

def Mann_Whitney_calc_s_star(data1, data2):
    m = len(data1) # размер первой выборки
    n = len(data2) # размер второй выборки
    u = Wilcoxon_calc_s_star(data1, data2)
    s_star = abs(u-n1*n2/2)/mth.sqrt(n1*n2*(n1+n2+1)/12) #статистика критерия
    return s_star

#предельное распределение Student v
def distr_t(s_star, v): 
    cdfunc = scst.t.cdf(s_star,v) 
    return cdfunc 
#предельное распределение norm
def distr_MW(s_star):
    cdfunc = scst.norm.cdf(s_star)
    return cdfunc

n1 = 20
n2 = 20
data1 = modeling(0, n1, xi=0, eta= 1)
data2 = modeling(0, n2, xi=0., eta=1)
mu1 = scst.norm.fit(data1)[0]
mu2 = scst.norm.fit(data2)[0]
s = Mann_Whitney_calc_s_star(data1, data2)
pv = 1-distr_MW(s)
#s = t_student_calc_s_star(data1, data2, mu1, mu2)
#моделирование статистики критерия
# def criterion_Anderson_Darling(mode, m, n, M):
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()
#     if size != 0:
#         proc_M = int(M / size)
#         data1_mod = np.array([modeling(mode, m) for i in range(proc_M)])
#         data2_mod = np.array([modeling(mode, n) for i in range(proc_M)])
#         s_n = np.array([AD_calc_s_star(np.sort(x), np.sort(y)) for x, y in zip(data1_mod, data2_mod)])
#         f =  np.array([distr_a2(x) for x in s_n])
#         s_n_new = np.zeros(M)
#         F = np.zeros(M)
#         comm.Gather(s_n, s_n_new, root=0) 
#         comm.Gather(f, F, root=0)
#         d = 0
#     if rank == 0:
#         np.savetxt('s_n_N(0., 1.)_40_42_2660000.dat', np.sort(s_n_new), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 40 42\n 0 2660000') 
#         ecdf = ECDF(s_n_new)
#         d = stats.ks_2samp(ecdf(np.sort(s_n_new)), F)
#         print("Done!")
#     MPI.Finalize()
#     return d