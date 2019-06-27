#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # Challenge: make your own regression model
# [Download the Excel file here](https://ucr.fbi.gov/crime-in-the-u.s/2013/crime-in-the-u.s.-2013/tables/table-8/table-8-state-cuts/table_8_offenses_known_to_law_enforcement_new_york_by_city_2013.xls) on crime data in New York State in 2013, provided by the FBI: UCR ([Thinkful mirror](https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/master/New_York_offenses/NEW_YORK-Offenses_Known_to_Law_Enforcement_by_City_2013%20-%2013tbl8ny.csv)).
# 
# Prepare this data to model with multivariable regression (including data cleaning if necessary) according to this specification:
# 
# $$ Property crime = \alpha + Population + Population^2 + Murder + Robbery$$
# 
# Now that you've spent some time playing with a sample multivariate linear regression model, it's time to make your own.
# 
# You've already gotten started by prepping the FBI:UCR Crime dataset (Thinkful mirror) in a previous assignment.
# 
# Using this data, build a regression model to predict property crimes. You can use the features you prepared in the previous assignment, new features of your own choosing, or a combination. The goal here is prediction rather than understanding mechanisms, so the focus is on creating a model that explains a lot of variance.
# 
# Submit a notebook with your model and a brief writeup of your feature engineering and selection process to submit and review with your mentor.

#%%
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("./datasets/NY-nKnowOffenses.csv")
df.columns = ['City', 'Population', 'Violent crime', 'Murder and nonnegligent manslaughter', 'Rape (revised definition)', 'Rape (legacy definition)', 'Robbery', 'Aggravated assault', 'Property crime', 'Burglary', 'Larceny theft', 'Motor vehicle theft', 'Arson']
df = df.iloc[4:]
df = df.iloc[:-3]
df


#%%
df['Property crime'] = df['Property crime'].str.replace(',','').astype('int')
df['Robbery'] = df['Robbery'].str.replace(',','').astype('int')
df['Population'] = df['Population'].str.replace(',','').astype('int')
df['Population_2'] = df['Population'].apply(lambda x: x * x)
df['Murder']= df['Murder and nonnegligent manslaughter'].astype('int')
df['RobberyClass'] = df['Robbery'].apply(lambda x: 1 if int(x) > 0 else 0)
df['MurderClass'] = df['Murder and nonnegligent manslaughter'].apply(lambda x: 1 if int(x) > 0 else 0)
#Drop the outlier
df = df.drop(df[df.Robbery == 19170].index)
df.describe()


#%%
# Make the correlation matrix.
corrmat = df.corr()
print(corrmat)

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(9, 6))

# Draw the heatmap using seaborn.
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

sns.scatterplot(x="Robbery", y="Property crime", data=df)
sns.set(rc={'figure.figsize':(20,8)})
plt.show()

#%% [markdown]
# ### As given in the task
# $$ Property crime = \alpha + Population + Population^2 + Murder + Robbery$$
# 

#%%
# Test your model with different holdout groups.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
Y = df['Property crime'].values.reshape(-1, 1)
X = df[['RobberyClass','MurderClass', 'Population','Population_2']]
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)


#%%
# Instantiate and fit our model.
regr = linear_model.LinearRegression()
#print(data['Sales'].values)

regr.fit(X_train, y_train)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
print('\nmean-squared:')
print(mean_squared_error(y_test, y_pred))


#%%
from sklearn.model_selection import cross_val_score
cross_val_score(regr, X, Y, cv=5)

#%% [markdown]
# #### Own feature engineering and selection
# 1. To get a better R-squared there was "motor vehicle theft" added, which is a property crime
# 2. Get rid of population squared, because the coefficient is very very tiny and therefore the feature has almost no impact.

#%%
# Instantiate and fit our model.
regr = linear_model.LinearRegression()
#print(data['Sales'].values)
Y = df['Property crime'].values.reshape(-1, 1)
X = df[['RobberyClass','MurderClass', 'Population', 'Motor vehicle theft']]
# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
regr.fit(X_train, y_train)


# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
print('\nmean-squared:')
print(mean_squared_error(y_test, y_pred))


#%%
print('Less overfitting:')
cross_val_score(regr, X, Y, cv=5)

#%% [markdown]
# ### With PCA

#%%
from sklearn.decomposition import PCA 
# Instantiate and fit our model.
regr = linear_model.LinearRegression()
#print(data['Sales'].values)
Y = df['Property crime'].values.reshape(-1, 1)
X = df[['RobberyClass','MurderClass', 'Population', 'Motor vehicle theft']]

sklearn_pca = PCA(n_components=3)
X_pca = sklearn_pca.fit_transform(X)

# Use train_test_split to create the necessary training and test groups
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=20)



regr.fit(X_train, y_train)


# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
print('\nmean-squared:')
print(mean_squared_error(y_test, y_pred))
print('\ncoss-val-score:')
cross_val_score(regr, X, Y, cv=5)


