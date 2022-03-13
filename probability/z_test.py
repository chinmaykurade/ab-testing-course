import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest
np.random.seed(0)

#%% One-Sample tests
N = 100
mu = 0.2
sigma = 1
x = np.random.randn(100)*sigma + mu

#%% Two-sided test
print(ztest(x))

#%% Two-sided test manual
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z = mu_hat / (sigma_hat / np.sqrt(N))
p_right = 1 - norm.cdf(abs(z))
p_left = norm.cdf(-abs(z))
p = p_right + p_left
print(z, p)

#%% One-sided test
ztest(x, alternative="larger")

#%% One-sided test manual
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z = mu_hat / (sigma_hat / np.sqrt(N))
p = 1-norm.cdf(abs(z))
print(z,p)

#%% REference value
ztest(x, value=0.05)

#%% REference value manual
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z = (mu_hat-0.05) / (sigma_hat / np.sqrt(N))
p_right = 1 - norm.cdf(abs(z))
p_left = norm.cdf(-abs(z))
p = p_right + p_left
print(z, p)


#%% Two-sample tests
N_0 = 1000
mu_0 = 0.2
sigma_0 = 1
x_0 = np.random.randn(N_0)*sigma_0 + mu_0

N_1 = 100
mu_1= 0.5
sigma_1 = 1
x_1 = np.random.randn(N_1)*sigma_1 + mu_1

#%%
ztest(x_0, x_1)

#%% Two-sided test manual
mu_hat_0 = x_0.mean()
mu_hat_1 = x_1.mean()
y = mu_hat_0 - mu_hat_1
s2_hat_0 = x_0.var(ddof=1)
s2_hat_1 = x_1.var(ddof=1)
s_hat = np.sqrt(s2_hat_1/N_1 + s2_hat_0/N_0)
z = y / s_hat
p_right = 1 - norm.cdf(abs(z))
p_left = norm.cdf(-abs(z))
p = p_right + p_left
print(z,p)


#%% False alarm (Type I error) 5% of times
result = []
for i in range(10_000):
    x0 = np.random.randn(100)
    x1 = np.random.randn(100)
    z,p = ztest(x0, x1)
    result.append(p<0.05)

print(np.mean(result))