import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2

#%%
df = pd.read_csv('data/advertisement_clicks.csv')
df.head()

#%% Create the contingency table
df_crosstab = pd.crosstab(df['advertisement_id'], df['action'], margins = False)

#%% chi2 api approach
stat, p, dof, expected = chi2_contingency(df_crosstab)
print(stat, p, dof)
if p<0.05:
    print("The result is significant. Reject Null hypothesis.")
else:
    print("Failed to reject Null hypothesis.")


#%% Chi2 code approach
column_sums = np.array(df_crosstab.sum(axis=0))
row_sums = np.array(df_crosstab.sum(axis=1))

expected = np.dot(row_sums.reshape(-1,1), column_sums.reshape(1,-1))/np.sum(row_sums)

diff = np.array(df_crosstab) - expected
numerator = diff ** 2
chi2_statistic = np.sum(numerator/expected)

print(chi2_statistic)


#%%
critical_value = chi2.ppf(0.95, df= 1)

p_value = chi2.sf(chi2_statistic, df=1)
# p_value = 1 - chi2.cdf(chi2_statistic, df=1)

if p_value<0.05:
    print("The result is significant. Reject Null hypothesis.")
else:
    print("Failed to reject Null hypothesis.")
