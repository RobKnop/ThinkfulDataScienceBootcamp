#%%
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass

import nltk
import pandas as pd
import pandas_profiling as pp
import numpy as np
pd.set_option('display.max_rows', 400)
pd.set_option('float_format', '{:.2f}'.format)
import re
import codecs
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
import IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats

#%%
df = pd.read_csv("data/ESSdata2_Thinkful.csv")
pp.ProfileReport(df).to_file("5.5.4_ChallengeStatsTests.html")

#%%
#Cleaning?
#df = df.dropna()
#%% [markdown]
# ## Task
# Using selected questions from the 2012 and 2014 editions of the European Social Survey, address the following questions. Keep track of your code and results in a Jupyter notebook or other source that you can share with your mentor. For each question, explain why you chose the approach you did.
# 
# Here is the data file. And here is the codebook, with information about the variable coding and content.
# 
# In this dataset, the same participants answered questions in 2012 and again 2014.
# 
# 1. Did people become less trusting from 2012 to 2014? Compute results for each country in the sample.
# 2. Did people become happier from 2012 to 2014? Compute results for each country in the sample.
# 3. Who reported watching more TV in 2012, men or women?
# 4. Who was more likely to believe people were fair in 2012, people living with a partner or people living alone?
# 5. Pick three or four of the countries in the sample and compare how often people met socially in 2014. Are there differences, and if so, which countries stand out?
# 6. Pick three or four of the countries in the sample and compare how often people took part in social activities, relative to others their age, in 2014. Are there differences, and if so, which countries stand out?
# 
##### Assumption: Every variable is normally-distributed  (see the histograms in the profile report)

#%%
# 1. Did people become less trusting from 2012 to 2014? Compute results for each country in the sample.

print(df.groupby(['year'])['ppltrst'].mean())

F, p = stats.f_oneway(
    df[df['year'] == 7].ppltrst.dropna(),
    df[df['year'] == 6].ppltrst.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))

#%% [markdown]
#### Answer: No difference in the trust score. So neither less nor higher. 
#%%
# 2. Did people become happier from 2012 to 2014? Compute results for each country in the sample.

print(df.groupby(['year'])['happy'].mean())

F, p = stats.f_oneway(
    df[df['year'] == 7].happy.dropna(),
    df[df['year'] == 6].happy.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))
#%% [markdown]
#### Answer: No, there is no statistically significant difference. 
#%%
# 3. Who reported watching more TV in 2012, men or women?

print(df.groupby(['gndr'])['tvtot'].mean())

F, p = stats.f_oneway(
    df[df['gndr'] == 1].tvtot.dropna(),
    df[df['gndr'] == 2].tvtot.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))
#%% [markdown]
#### Answer: No, there is no statistically significant difference. 
#%%
# 4. Who was more likely to believe people were fair in 2012, 
# people living with a partner or people living alone?
df_4 = df[df['year'] == 6] # filter for 2012

print(df_4.groupby(['partner'])['pplfair'].mean())


F, p = stats.f_oneway(
    df_4[df_4['partner'] == 1].pplfair.dropna(),
    df_4[df_4['partner'] == 2].pplfair.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))
#%% [markdown]
#### Answer: Who is living with a partner
#%%
# 5. Pick three or four of the countries in the sample and compare how often people met socially in 2014. 
# Are there differences, and if so, which countries stand out?
df_5 = df[df['year'] == 7] # filter for 2014

print(df_5.groupby(['cntry'])['sclact'].mean())


F, p = stats.f_oneway(
    df_5[df_5['cntry'] == 'CH'].sclact.dropna(),
    df_5[df_5['cntry'] == 'SE'].sclact.dropna(),
    df_5[df_5['cntry'] == 'ES'].sclact.dropna(),
    df_5[df_5['cntry'] == 'NO'].sclact.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))
#%% [markdown]
#### Answer: There is a statistically significant difference between countries. Sweden stands out.
#%%
# 6. Pick three or four of the countries in the sample and compare how often people took part in social activities, 
# relative to others their age, in 2014. 
# Are there differences, and if so, which countries stand out?
df_6 = df[df['year'] == 7] # filter for 2014

print(df_6.groupby(['cntry', 'sclact'])['agea'].mean())

print('CH:')
df_CH = df_6[df_6['cntry'] == 'CH']

F, p = stats.f_oneway(
    df_CH[df_CH['sclact'] == 1].agea.dropna(),
    df_CH[df_CH['sclact'] == 2].agea.dropna(),
    df_CH[df_CH['sclact'] == 3].agea.dropna(),
    df_CH[df_CH['sclact'] == 4].agea.dropna(),
    df_CH[df_CH['sclact'] == 5].agea.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))

print('NO:')
df_NO = df_6[df_6['cntry'] == 'NO']

F, p = stats.f_oneway(
    df_NO[df_NO['sclact'] == 1].agea.dropna(),
    df_NO[df_NO['sclact'] == 2].agea.dropna(),
    df_NO[df_NO['sclact'] == 3].agea.dropna(),
    df_NO[df_NO['sclact'] == 4].agea.dropna(),
    df_NO[df_NO['sclact'] == 5].agea.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))

print('SE:')
df_SE = df_6[df_6['cntry'] == 'SE']

F, p = stats.f_oneway(
    df_SE[df_SE['sclact'] == 1].agea.dropna(),
    df_SE[df_SE['sclact'] == 2].agea.dropna(),
    df_SE[df_SE['sclact'] == 3].agea.dropna(),
    df_SE[df_SE['sclact'] == 4].agea.dropna(),
    df_SE[df_SE['sclact'] == 5].agea.dropna()
)

# The F statistic.
print('F value: ', F)

# The probability. A p < .05 would lead us to believe the group means were
# not all similar in the population.
print("p-value : %0.6f " % (p))

#%% [markdown]
#### Answer: There is a statistically significant difference between ages. No one stands out.