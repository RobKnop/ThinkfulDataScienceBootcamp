#%% [markdown]
# # Housing Prices
# Using this Kaggle data create a model to predict a house's value. We want to be able to understand what creates value in a house, as though we were a real estate developer.
#%%
import os
from IPython import get_ipython
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
import pandas_profiling as pp
import numpy as np
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Load models
from sklearn import ensemble, tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD 
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
#%%
# Source: https://www.kaggle.com/anthonypino/melbourne-housing-market/downloads/melbourne-housing-market.zip/27
df = pd.read_csv("Main/data/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv")
pp.ProfileReport(df, check_correlation=True).to_file(outputfile="3.6.4_ProfileOfHousingPrices_RAW.html")
#%% [markdown]
# ### Variable descriptions
# * Suburb: Suburb
# * Address: Address
# * Rooms: Number of rooms
# * Price: Price in Australian dollars
# * Method: S - property sold; SP - property sold prior; PI - property passed in; PN - sold prior not disclosed; SN - sold not disclosed; NB - no bid; VB - vendor bid; W - withdrawn prior to auction; SA - sold after auction; SS - sold after auction price not disclosed. N/A - price or highest bid not available.
# * Type: br - bedroom(s); h - house,cottage,villa, semi,terrace; u - unit, duplex; t - townhouse; dev site - development site; o res - other residential.
# * SellerG: Real Estate Agent
# * Date: Date sold
# * Postcode
# * Distance: Distance from CBD in Kilometres
# * Regionname: General Region (West, North West, North, North east ...etc)
# * Propertycount: Number of properties that exist in the suburb.
# * CouncilArea: Governing council for the area

#%%
# Drop ca. 2% of rows which have a lot of missing values
df = df.dropna(subset=['Price'])
# Drop unnecessary columns
df = df.drop(columns=[
    'Address', # too many distinct values
    'Date' # not necessary 
])
# Drop duplicated
df = df.drop_duplicates() # Dataset has 2 duplicate rows

# Do second data profile report on cleaned data
pp.ProfileReport(df, check_correlation=False, pool_size=15).to_file(outputfile="3.6.4_ProfileOfHousingPrices_CLEAN.html")
# See the webpage at: https://github.com/RobKnop/ThinkfulDataScienceBootcamp/blob/master/Main/3.6.4_ProfileOfHousingPrices_CLEAN_5mio.html

# Make the correlation matrix.
corrmat = df.corr()
print(corrmat)

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(9, 6))

# Draw the heatmap using seaborn.
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

#%% [markdown]
# #### Findings
# 1. Correlation to y (delayed) exists: 
# 2. Multicollinearity is in general low, but certain variables are highly correlated
#   * like DEP_xxxx vars
#   * DISTANCE - DISTANCE_GROUP - AIR_TIME
# 3. Class imbalance: 17601697 - 2398303 (88%/12%)
#%% [markdown]
# #### Our key evaluation metric to optimize on is accuracy, followed by the f1 score 
# * A balance between precision and recall is needed. 
#%% [markdown]
# #### Models to try:
# 1. Linear Regression
# 2. Naive Bayes 
# 4. RandomForestRegressir
# 5. KNN
# 6. Support Vector Machine
# 7. GradientBoosting Regression 
# 8. (Also use of KSelectBest, GridSearch)
#%%
# Normalize
""" mm_scaler = MinMaxScaler()
df[['Distance']] = mm_scaler.fit_transform(df[['Distance']].values)
df[['Amount']] = mm_scaler.fit_transform(df[['Amount']].values) """

# Define X and y
X = df.drop(columns=[
                    'Price', # is the Y
                    'Suburb', # is categorical 
                    'SellerG', # is categorical 
                    'Type', # is categorical 
                    'Regionname', # is categorical 
                    'Method', # is categorical 
                    'CouncilArea' # is categorical 
                    ])
X = pd.concat([X, pd.get_dummies(df['Suburb'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['SellerG'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['Type'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['Regionname'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['Method'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['CouncilArea'])], axis=1)

y = df['Price']

#Try SelectKBest
X_selKBest = SelectKBest(k=300).fit_transform(X, y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_selKBest, y, test_size=0.2, random_state=20)
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

score = cross_val_score(regr, X, y, cv=5)
print("Cross Validated Score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%% 
for k in range(5, 39, 1):
  print('\nk = ', k)
  knn = KNeighborsRegressor(n_neighbors=k)
  knn.fit(X_train, y_train)
  print('KNN R^2 score: ', knn.score(X_test, y_test))

  knn_w = KNeighborsRegressor(n_neighbors=k, weights='distance')
  knn_w.fit(X_train, y_train)
  print('KNN_dist R^2 score: ', knn_w.score(X_test, y_test))

"""
Best k =  7
KNN R^2 score:  0.7175245604677205
KNN_dist R^2 score:  0.6813617988109184
"""
#%%
k = 7
score = cross_val_score(KNeighborsRegressor(n_neighbors=k), X, y, cv=5)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score_w = cross_val_score(KNeighborsRegressor(n_neighbors=k, weights='distance'), X, y, cv=5)
print("Weighted Accuracy: %0.2f (+/- %0.2f)" % (score_w.mean(), score_w.std() * 2))
#%% [markdown]
# #### Final model evaluation:
# The best model 

#### Other models
