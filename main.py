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
    return s_star

def freedom_degree(data1, data2):
    n1 = len(data1) # размер первой выборки
    n2 = len(data2) # размер второй выборки
    s1 = data1.var()
    s2 = data2.var()
    if n1 != n2:
        v = (s1 / n1 + s2 / n2)**2 / ((s1 / n1)**2 / (n1 - 1)+ (s2 / n2)**2 / (n2 - 1))
    else:
        v = n1 - 1 + (2 * n1 - 2) / (s1**2 / s2**2 + s2**2 / s1**2)
    return v

def Wilcoxon_calc_s_star(data1, data2):
    n1 = len(data1) # размер первой выборки
    n2 = len(data2) # размер второй выборки
    data = np.sort(np.concatenate((data1, data2))) # объединенная выборка
    r1 = np.sum(list(map(lambda x: x + 1, np.where(np.isin(data, data1))[0])))
    r2 = np.sum(list(map(lambda x: x + 1, np.where(np.isin(data, data2))[0])))
    u1 = n1 * n2 + n1 * (n1 + 1) / 2 - r1
    u2 = n1 * n2 + n2 * (n2 + 1) / 2 - r2
    s_star =  min(u1, u2) #статистика критерия
    return s_star

def Mann_Whitney_calc_s_star(data1, data2):
    m = len(data1) # размер первой выборки
    n = len(data2) # размер второй выборки
    u = Wilcoxon_calc_s_star(data1, data2)
    s_star = abs(u - n1 * n2 / 2) / mth.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) #статистика критерия
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
    alpha = 0.05
    data1_mod1 = np.array([modeling(0, n1, xi = 0, eta = 1) for i in range(M)])
    data2_mod1 = np.array([modeling(0, n2, xi = 0, eta = 1.1) for i in range(M)])
    s_n1 = np.array([Mann_Whitney_calc_s_star(x, y) for x, y in zip(data1_mod1, data2_mod1)])#np.array([t_student_calc_s_star(np.sort(x), np.sort(y)) for x, y in zip(data1_mod1, data2_mod1)])

    data1_mod2 = np.array([modeling(0, n1, xi = 0, eta = 1) for i in range(M)])
    data2_mod2 = np.array([modeling(0, n2, xi = 0.01, eta = 1.1) for i in range(M)])
    s_n2 = np.array([Mann_Whitney_calc_s_star(x, y) for x, y in zip(data1_mod2, data2_mod2)])#np.array([t_student_calc_s_star(np.sort(x), np.sort(y)) for x, y in zip(data1_mod2, data2_mod2)])
    #v = np.array([freedom_degree(x,y) for x, y in zip(data1_mod2, data2_mod2)]).mean()
    s_a_2 = scst.norm.ppf(alpha/2)#scst.t.ppf(alpha/2, v)
    s_1_a_2 = scst.norm.ppf(1-alpha/2)#scst.t.ppf(1-alpha/2, v)
    ecdf = ECDF(s_n2)
    p1_beta = 1 - (ecdf(s_1_a_2) - ecdf(s_a_2))
    np.savetxt('s_n_N(0., 1.)_1_1000.dat', np.sort(s_n1), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 1\n 0 16600') 
    np.savetxt('s_n_N(0., 1.)_2_1000.dat', np.sort(s_n2), fmt='%.14f', delimiter=' ', newline='\n', header='S_n 2\n 0 16600') 
    return s_n1

n1 = 35
n2 = 35
criterion_stat_modeling(n1, n2, 16600)

# s = Mann_Whitney_calc_s_star(data1, data2)
# pv = 1-distr_MW(s)
#s = t_student_calc_s_star(data1, data2, mu1, mu2)

