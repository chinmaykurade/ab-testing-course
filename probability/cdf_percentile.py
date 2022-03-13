import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(0)

#%% https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
# Generate the sample
mu = 170
sd = 8
x = norm.rvs(loc=mu, scale=sd, size=100)

#%% Statistics
print(x.mean())
print(x.var())
print(x.std())

#%% Delta Degrees of freedom
print(x.var(ddof=1))
print(x.std(ddof=1))

#%% Quantile function (inverse of cdf)
print(norm.ppf(0.95, loc=mu, scale=sd))

#%% CDF
print(norm.cdf(165, loc=mu, scale=sd))

#%% Survival Function = 1 - cdf
print(1-norm.cdf(165, loc=mu, scale=sd))
print(norm.sf(165, loc=mu, scale=sd))
