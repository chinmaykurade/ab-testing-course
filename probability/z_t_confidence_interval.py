import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
np.random.seed(1)

#%%
N = 1_0
mu = 5
sd = 2

#%%
x = np.random.randn(N)*sd + mu

#%% Z-CI
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
z_left = norm.ppf(0.0250)
z_right = norm.ppf(0.9750)
left_ci = mu_hat + z_left*sigma_hat/np.sqrt(N)
right_ci = mu_hat + z_right*sigma_hat/np.sqrt(N)
print(f"Mean: {mu_hat}")
print(f"Confidence Interval is [{left_ci}, {right_ci}]")

#%% t-CI
mu_hat = x.mean()
sigma_hat = x.std(ddof=1)
t_left = t.ppf(0.0250, df=N-1)
t_right = t.ppf(0.9750, df=N-1)
left_ci = mu_hat + t_left*sigma_hat/np.sqrt(N)
right_ci = mu_hat + t_right*sigma_hat/np.sqrt(N)
print(f"Mean: {mu_hat}")
print(f"Confidence Interval is [{left_ci}, {right_ci}]")

#%%
def experiment():
    x = np.random.randn(N) * sd + mu
    mu_hat = x.mean()
    sigma_hat = x.std(ddof=1)
    t_left = t.ppf(0.0250, df=N - 1)
    t_right = t.ppf(0.9750, df=N - 1)
    left_ci = mu_hat + t_left * sigma_hat / np.sqrt(N)
    right_ci = mu_hat + t_right * sigma_hat / np.sqrt(N)
    return mu<right_ci and mu>left_ci

def multi_experiment(n):
    results = [experiment() for _ in range(n)]
    return np.mean(results)

#%%
result = multi_experiment(10_000)
print(result)
