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
def t_student_calc_s_star(n1, n2, data1, data2):
    x1_mean = data1.mean()
    x2_mean = data2.mean()
    s1 = data1.var(ddof=1)
    s2 = data2.var(ddof=1)
    s_star = (x1_mean - x2_mean) / (mth.sqrt(s1 / n1 + s2 / n2)) #статистика критерия
    return s_star

def freedom_degree(n1, n2, data1, data2):
    s1 = data1.var()
    s2 = data2.var()
    if n1 != n2:
        v = (s1 / n1 + s2 / n2)**2 / ((s1 / n1)**2 / (n1 - 1)+ (s2 / n2)**2 / (n2 - 1))
    else:
        v = n1 - 1 + (2 * n1 - 2) / (s1**2 / s2**2 + s2**2 / s1**2)
    return v

def Wilcoxon_calc_s_star(n1, n2, data1, data2):
    data = np.sort(np.concatenate((data1, data2))) # объединенная выборка
    t = np.sum(list(map(lambda x: x + 1, np.where(np.isin(data, data2))[0])))
    u = n1 * n2 + (n2 * (n2 + 1)) / 2 - t
    return u

def Mann_Whitney_calc_s_star(n1, n2, data1, data2):
    u = Wilcoxon_calc_s_star(n1, n2, np.sort(data1), np.sort(data2))
    divv = mth.sqrt(((n1 * n2) * (n1 + n2 + 1)) / 12.)
    s_star = (u - (n1 * n2) / 2.) / divv #статистика критерия
    return s_star

#предельное распределение Student v
def distr_t(s_star, v): 
    cdfunc = scst.t.cdf(s_star, v) 
    return cdfunc 

#предельное распределение norm
def distr_MW(s_star):
    cdfunc = scst.norm.cdf(s_star)
    return cdfunc

#моделирование статистики критерия
def criterion_stat_modeling(n1, n2, M):
    alpha = [0.01,0.025,0.05,0.1,0.15]
    data1_mod1 = np.array([np.random.normal(loc=0, scale=1, size=n1) for i in range(M)])
    data2_mod1 = np.array([np.random.normal(loc=0, scale=1, size=n2) for i in range(M)])
    s_n1 = np.array([Mann_Whitney_calc_s_star(n1, n2, x, y) for x, y in zip(data1_mod1, data2_mod1)])

    data1_mod2 = np.array([np.random.normal(loc=0, scale=1, size=n1) for i in range(M)])
    data2_mod2 = np.array([np.random.normal(loc=0, scale=10, size=n2) for i in range(M)])
    s_n2 =np.array([Mann_Whitney_calc_s_star(n1, n2, x, y) for x, y in zip(data1_mod2, data2_mod2)]) 

    #v = np.array([freedom_degree(x,y) for x, y in zip(data1_mod2, data2_mod2)]).mean()
    s_a_2 = [scst.norm.ppf(x/2) for x in alpha]#scst.t.ppf(alpha/2, v)
    s_1_a_2 = [scst.norm.ppf(1-x/2) for x in alpha]#scst.t.ppf(1-alpha/2, v)
    ecdf2 = ECDF(s_n2)
    ecdf1 = ECDF(s_n1)
    d = scst.ks_2samp(ecdf1(np.sort(s_n1)), ecdf2(np.sort(s_n2)))
    p1 = [1 - (ecdf2(s_1_a_2[i]) - ecdf2(s_a_2[i])) for i in range(len(alpha))]
    #np.savetxt('s_n_N(0., 1.)_1_5.dat', np.sort(data1_mod1), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 5\n 0 5') 
    #np.savetxt('s_n_N(0., 1.)_2_5.dat', np.sort(data1_mod1), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 5\n 0 5') 
    np.savetxt('s_n_N(0., 1.)_5_16600.dat', np.sort(s_n1), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 5\n 0 16600') 
    np.savetxt('s_n_N(0., 1.)_30_16600.dat', np.sort(s_n2), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 30\n 0 16600') 
    np.savetxt('power.txt', p1, fmt='%.4f', delimiter=' ', newline='\n')
    return s_n1

n1 = 100
n2 = 100
s_n1 = criterion_stat_modeling(n1, n2, 16600)

# s = Mann_Whitney_calc_s_star(data1, data2)
# pv = 1-distr_MW(s)
#s = t_student_calc_s_star(data1, data2, mu1, mu2)

