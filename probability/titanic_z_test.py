# Perform z-test to check if mean fare for survived vs non-survived is different
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import ztest

#%%
df = pd.read_csv(r'data/train.csv')
df.head()

#%%
x1 = df.query('Survived == 1')['Fare']
x2 = df.query('Survived == 0')['Fare']

#%%
sns.kdeplot(x1, label="Survived")
sns.kdeplot(x2, label="Did not survive")
plt.legend()
plt.show()


#%%
x1.mean(), x2.mean()

#%%
ztest(x1, x2)