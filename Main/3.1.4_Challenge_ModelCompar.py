#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Challenge Model Comparison
# You now know two kinds of regression and two kinds of classifier. So let's use that to compare models!
# 
# Comparing models is something data scientists do all the time. There's very rarely just one model that would be possible to run for a given situation, so learning to choose the best one is very important.
# 
# Here let's work on regression. Find a data set and build a KNN Regression and an OLS regression. Compare the two. How similar are they? Do they miss in different ways?
# 
# Create a Jupyter notebook with your models. At the end in a markdown cell write a few paragraphs to describe the models' behaviors and why you favor one model or the other. Try to determine whether there is a situation where you would change your mind, or whether one is unambiguously better than the other. Lastly, try to note what it is about the data that causes the better model to outperform the weaker model. Submit a link to your notebook below.

#%%
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Source : https://ucr.fbi.gov/crime-in-the-u.s/2013/crime-in-the-u.s.-2013/tables/table-8/table-8-state-cuts/table_8_offenses_known_to_law_enforcement_new_york_by_city_2013.xls
df = pd.read_csv("../Unit2/datasets/NY-nKnowOffenses.csv")
df.columns = ['City', 'Population', 'Violent crime', 'Murder and nonnegligent manslaughter', 'Rape (revised definition)', 'Rape (legacy definition)', 'Robbery', 'Aggravated assault', 'PropertyCrime', 'Burglary', 'Larceny theft', 'Motor vehicle theft', 'Arson']
df = df.iloc[4:]
df = df.iloc[:-3]
df['PropertyCrime'] = df['PropertyCrime'].str.replace(',','').astype('int')
df['Robbery'] = df['Robbery'].str.replace(',','').astype('int')
df['Population'] = df['Population'].str.replace(',','').astype('int')
df['Population_2'] = df['Population'].apply(lambda x: x * x)
df['Murder']= df['Murder and nonnegligent manslaughter'].astype('int')
df['Motor_vehicle_theft'] = df['Motor vehicle theft'].str.replace(',','').astype('int')
df['RobberyClass'] = df['Robbery'].apply(lambda x: 1 if int(x) > 0 else 0)
df['MurderClass'] = df['Murder and nonnegligent manslaughter'].apply(lambda x: 1 if int(x) > 0 else 0)
df['Burglary'] = df['Burglary'].str.replace(',','').astype('int')
#Drop the outlier
df = df.drop(df[df.Robbery == 19170].index)
df.describe()

#%% [markdown]
# ### OLS Model

#%%
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.formula.api as smf
import statsmodels.api as sm
 
class statsmodel(BaseEstimator, RegressorMixin):
    def __init__(self, sm_class, formula):
        self.sm_class = sm_class
        self.formula = formula
        self.model = None
        self.result = None
 
    def fit(self,data,dummy):
        self.model = self.sm_class(self.formula,data)
        self.result = self.model.fit()
 
    def predict(self,X):
        return self.result.predict(X)


#%%
Y = df['PropertyCrime']#.values.reshape(-1, 1)
X = df[['RobberyClass','MurderClass', 'Population','Motor_vehicle_theft']]
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

df_train = X_train.copy()
df_train['PropertyCrime'] = y_train.copy()

linear_formula = 'PropertyCrime ~ RobberyClass+MurderClass+Population+Motor_vehicle_theft'

# create a model
clf = statsmodel(smf.ols, linear_formula)
# fit a model
clf.fit(df_train, None)


# Inspect the results.
print('\nParameters: \n', clf.result.params)
print('\np-values: \n', clf.result.pvalues)
print('\nR-squared:')
print(clf.result.rsquared)
print('\nConfidence intervals:')
print(clf.result.conf_int())

y_pred = clf.predict(X_test)
print('\nmean-squared:')
print(mean_squared_error(y_test, y_pred))

print('\nLess overfitting:')
print('coss-val-score:')
score = cross_val_score(clf, df, df['PropertyCrime'], cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))


#%%
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from scipy import stats

for i in range(20):
    
    print('\nk = ', i+1)
    
    X = df[['RobberyClass','MurderClass', 'Population','Motor_vehicle_theft']]
    X = pd.DataFrame({
    'RobberyClass': stats.zscore(df.RobberyClass),
    'MurderClass': stats.zscore(df.MurderClass),
    'Population' : stats.zscore(df.Population),
    'Motor_vehicle_theft': stats.zscore(df.Motor_vehicle_theft)
    })
    Y = df['PropertyCrime']
    knn = neighbors.KNeighborsRegressor(n_neighbors=i+1)
    knn.fit(X, Y)
    
    knn_w = neighbors.KNeighborsRegressor(n_neighbors=i+1, weights='distance')
    knn_w.fit(X, Y)


    score = cross_val_score(knn, X, Y, cv=5)
    print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    score_w = cross_val_score(knn_w, X, Y, cv=5)
    print("Weighted Accuracy: %0.2f (+/- %0.2f)" % (score_w.mean(), score_w.std() * 2))

#%% [markdown]
# ## Evaluation
# From KNN I going to chosse the model with k = 9 <br>
# Unweighted Accuracy: 0.62 (+/- 0.16)<br>
# 
# This one has the best balance between accuracy and standard deviation.
#%% [markdown]
# ### Comparision KNN and OLS
# OLS --> Accuracy: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.78 (+/- 0.37)<br>
# KNN --> Unweighted Accuracy: 0.62 (+/- 0.16)<br>
# 
# Recommendation: Choose the OLS, because even if the KNN has  0.62 + 0.16 = 0.78 accuracy and that is still under the mean accuracy of the OLS model
# But there is no one unambiguously better model in this notebook.
#%% [markdown]
# ### What it is about the data that causes the better model to outperform the weaker model.

#%%
# Make the correlation matrix.
corrmat = df.corr()
# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(9, 6))

# Draw the heatmap using seaborn.
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#%% [markdown]
# Probably because of the high correlation of the inputs to the output (property crime) the OLS model is working better than the clustering approach of KNNs. 

