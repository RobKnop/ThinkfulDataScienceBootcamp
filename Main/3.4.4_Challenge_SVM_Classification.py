#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Challenge
# Oh dear, so this did seem not to work very well. In fact it is remarkably poor. Now there are many things that we could do here. 
# 
# Firstly the overfit is a problem, even though it was poor in the first place. We could go back and clean up our feature set. There might be some gains to be made by getting rid of the noise.
# 
# We could also see how removing the nulls but including dietary information performs. Though its a slight change to the question we could still possibly get some improvements there.
# 
# Lastly, we could take our regression problem and turn it into a classifier. With this number of features and a discontinuous outcome, we might have better luck thinking of this as a classification problem. We could make it simpler still by instead of classifying on each possible value, group reviews to some decided high and low values.
# 
# __And that is your challenge.__
# 
# Transform this regression problem into a binary classifier and clean up the feature set. You can choose whether or not to include nutritional information, but try to cut your feature set down to the 30 most valuable features.
# 
# Good luck!

#%%
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


#%%
raw_data = pd.read_csv('https://tf-assets-prod.s3.amazonaws.com/tf-curric/data-science/epi_r.csv')


#%%
#list(raw_data.columns)


#%%
raw_data.sample(5)


#%%
raw_data.rating.describe()


#%%
raw_data.rating.hist(bins=20)
plt.title('Histogram of Recipe Ratings')
plt.show()


#%%
# Count nulls 
null_count = raw_data.isnull().sum()
null_count[null_count>0]


#%%
from sklearn.svm import SVR
svr = SVR()
X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1).sample(frac=0.3, replace=True, random_state=1)
Y = raw_data.rating.sample(frac=0.3, replace=True, random_state=1)


#%%
from sklearn.model_selection import cross_val_score
cross_val_score(svr, X, Y, cv=5)


#%%
# Transform rating into categorical variable
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 0.0, raw_data['rating'] < 1.0 ), 0, raw_data['rating'])
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 1.0, raw_data['rating'] < 2.0 ), 1, raw_data['rating'])
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 2.0, raw_data['rating'] < 3.0 ), 2, raw_data['rating'])
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 3.0, raw_data['rating'] < 4.0 ), 3, raw_data['rating'])
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 4.0, raw_data['rating'] < 5.0 ), 4, raw_data['rating'])
raw_data['rating'] = np.where(np.logical_and(raw_data['rating'] >= 5.0, raw_data['rating'] < 6.0 ), 5, raw_data['rating'])
raw_data['rating'].head(5)


#%%
from sklearn.svm import SVC
svc = SVC()
X = raw_data.drop(['rating', 'title', 'calories', 'protein', 'fat', 'sodium'], 1)
Y = raw_data.rating


#%%
cross_val_score(svc, X, Y, cv=5)

#%% [markdown]
# When you've finished that, also take a moment to think about bias. Is there anything in this dataset that makes you think it could be biased, perhaps extremely so?
# 
# There is. Several things in fact, but most glaringly is that we don't actually have a random sample. It could be, and probably is, that the people more likely to choose some kinds of recipes are more likely to give high reviews.
# 
# After all, people who eat chocolate _might_ just be happier people.

