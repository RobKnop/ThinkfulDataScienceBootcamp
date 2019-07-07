#%% [markdown]
# # Capstone Project
# You're ready to put into practice everything you've learned so far.
#
#First: Go out and find a dataset of interest. It could be from one of our recommended resources, some other aggregation, or scraped yourself. Just make sure it has lots of variables in it, including an outcome of interest to you.
#
#Second: Explore the data. Get to know the data. Spend a lot of time going over its quirks and peccadilloes. You should understand how it was gathered, what's in it, and what the variables look like.
#
#Third: Model your outcome of interest. You should try several different approaches and really work to tune a variety of models before using the model evaluation techniques to choose what you consider to be the best performer. Make sure to think about explanatory versus predictive power and experiment with both.
#
#So, here is the deliverable: Prepare a slide deck and 15 minute presentation that guides viewers through your model. Be sure to cover a few specific things:
#
#A specified research question your model addresses
#How you chose your model specification and what alternatives you compared it to
#The practical uses of your model for an audience of interest
#Any weak points or shortcomings of your model
#This presentation is not a drill. You'll be presenting this slide deck live to a group as the culmination of your work in the last two supervised learning units. As a secondary matter, your slides and / or the Jupyter notebook you use or adapt them into should be worthy of inclusion as examples of your work product when applying to jobs.

#%%
import os
from IPython import get_ipython
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
import pandas_profiling as pp
import numpy as np
import datetime as dt   
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Load models
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.naive_bayes import BernoulliNB

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.decomposition import PCA 

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#%%
# Source: https://www.kaggle.com/brittabettendorf/berlin-airbnb-data
df = pd.read_csv("listings.csv")
## pp.ProfileReport(df, check_correlation=True).to_file(outputfile="Main/3.7_ProfileOfBerlinAirBnB_RAW.html")
# See the webpage at: https://github.com/RobKnop/ThinkfulDataScienceBootcamp/blob/master/Main/3.7_ProfileOfBerlinAirBnB_RAW.html
#%% [markdown]
# ### Variable descriptions
# * id
# * name
# * host_id
# * host_name
# * neighbourhood_group
# * neighbourhood
# * latitude
# * longitude
# * room_type
# * price
# * minimum_nights
# * number_of_reviews
# * last_review
# * reviews_per_month
# * calculated_host_listings_count
# * availability_365

#%%
# Feature Engineering and Selection (Round 1)
df['last_review'] = pd.to_datetime(df['last_review'])
df['days_since_last_review'] = (df['last_review'] - dt.datetime(2018, 12, 31)).dt.days
df['y_price'] = df['price']

# Drop unnecessary columns
df = df.drop(columns=[
    'name', # we won't do any NLP here
    'last_review', # already converted into 'days_since_last_review'
    'price', # was copied into 'y_price'
    'id', # just a increasing number
])

# Cleaning: Get rid of outliers
# Drop examples where 
# the price is higher than 500€ (0.1% of all data)
# and lower than 10€
df = df[df['y_price'] > 10] # 22522 - 22491 = 31 --> under 0.1% of all data
df = df[df['y_price'] < 500] # 22491 - 22405 = 86 --> under 0.4% of all data

#%% 
plt.figure(figsize=(30, 20))

df.sort_values(by=['minimum_nights'])
plt.subplot(2, 3, 1)
plt.scatter(df['minimum_nights'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('Minimum nights to stay')

df.sort_values(by=['number_of_reviews'])
plt.subplot(2, 3, 2)
plt.scatter(df['number_of_reviews'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('Number of reviews')

df.sort_values(by=['calculated_host_listings_count'])
plt.subplot(2, 3, 3)
plt.scatter(df['calculated_host_listings_count'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('Host listings count')

df.sort_values(by=['availability_365'])
plt.subplot(2, 3, 4)
plt.scatter(df['availability_365'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('All year availability')

df.sort_values(by=['days_since_last_review'])
plt.subplot(2, 3, 5)
plt.scatter(df['days_since_last_review'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('Days since last review')

df.sort_values(by=['reviews_per_month'])
plt.subplot(2, 3, 6)
plt.scatter(df['reviews_per_month'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('Review per month')
plt.savefig('Main/3.7_Viz_Numeric_Features.png', dpi=100)
plt.close()
#%%
plt.scatter(df['latitude'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('latitude')
plt.show()
plt.scatter(df['longitude'], df['y_price'], color='red')
plt.ylim([0, max(df['y_price']) + 100])
plt.ylabel('price in €')
plt.title('longitude')
plt.show()
#%%
# Cleaning: Fill NaNs
values_to_fill = {
    'days_since_last_review': df.days_since_last_review.mean(), 
    'reviews_per_month': 0,
    'latitude': df.latitude.mean(),
    'longitude': df.longitude.mean()
    }
df = df.fillna(value=values_to_fill)

# Do second data profile report on cleaned data
pp.ProfileReport(df, check_correlation=True, pool_size=15).to_file(outputfile="Main/3.7_ProfileOfBerlinAirBnB_CLEAN.html")
# See the webpage at: https://github.com/RobKnop/ThinkfulDataScienceBootcamp/blob/master/Main/3.7_ProfileOfBerlinAirBnB_CLEAN.html

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
# 1. Correlation to y (price) is low: 
# 2. Multicollinearity is low
#%% [markdown]
# #### Our key evaluation metric to optimize on is root mean squared error
#%% [markdown]
# #### Models to try:
# 1. Linear Regression
# 4. RandomForestRegressor
# 5. KNN
# 6. Support Vector Machine
# 7. GradientBoosting Regression 
# 8. (Also use of KSelectBest, GridSearch, PCA)
#%%
# Normalize
mm_scaler = MinMaxScaler()
df[['longitude']] = mm_scaler.fit_transform(df[['longitude']].values)
df[['latitude']] = mm_scaler.fit_transform(df[['latitude']].values)
df[['minimum_nights']] = mm_scaler.fit_transform(df[['minimum_nights']].values)
df[['number_of_reviews']] = mm_scaler.fit_transform(df[['number_of_reviews']].values)
df[['reviews_per_month']] = mm_scaler.fit_transform(df[['reviews_per_month']].values)
df[['availability_365']] = mm_scaler.fit_transform(df[['availability_365']].values)
df[['calculated_host_listings_count']] = mm_scaler.fit_transform(df[['calculated_host_listings_count']].values)
df[['days_since_last_review']] = mm_scaler.fit_transform(df[['days_since_last_review']].values)

# Define X and y
X = df.drop(columns=[
                    'y_price', # is the Y
                    'neighbourhood_group', # is categorical 
                    'neighbourhood', # is categorical 
                    'room_type', # is categorical 
                    'host_id', # to heavy now, for later evaluation 
                    'host_name' # to heavy now, for later evaluation
                    ])
X = pd.concat([X, pd.get_dummies(df['neighbourhood_group'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['neighbourhood'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['room_type'])], axis=1)

y = df['y_price']

#Try SelectKBest
# X_selKBest = SelectKBest(k=120).fit_transform(X, y)

# Use PCA (but it is not working better)
# sklearn_pca = PCA(n_components=100)
# X_pca = sklearn_pca.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
# Linear Regression: Instantiate and fit our model.
regr = linear_model.LinearRegression()
#print(data['Sales'].values)

regr.fit(X_train, y_train)

# Inspect the results.
y_pred = regr.predict(X_test)
print('\nmean-squared:', mean_squared_error(y_test, y_pred))
rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))
print('R^2 score: ', regr.score(X_test, y_test)) 
'''
Plain:
    mean-squared: 1355.286586251543
    rms error is: 36.81421717559051
    R^2 score:  0.2718589837419886
    Cross Validated Score: -21932916364088741888.00 (+/- 87711703710492344320.00)
SelectKBest:

PCA:
    mean-squared: 1352.2922436354918
    rms error is: 36.77352639651918
    R^2 score:  0.2734677236923385
    Cross Validated Score: -5620486015875508338688.00 (+/- 22481944014540442173440.00)
'''
# Cross validate
score = cross_val_score(regr, X, y, cv=5, n_jobs=-1, verbose=1)
print("Cross Validated Score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%% 
# KNN:
for k in range(24, 30, 1):
    print('\nk = ', k)
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('mean-squared:', mean_squared_error(y_test, y_pred))
    rmse_val = rmse(y_pred, y_test)
    print("rms error is: " + str(rmse_val))
    print('KNN R^2 score: ', knn.score(X_test, y_test))
    
    knn_w = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_w.fit(X_train, y_train)
    y_pred = knn_w.predict(X_test)
    print('\nmean-squared:', mean_squared_error(y_test, y_pred))
    rmse_val = rmse(y_pred, y_test)
    print("rms error is: " + str(rmse_val))
    print('KNN_dist R^2 score: ', knn_w.score(X_test, y_test))
#%%
# Run best Model
k = 24
knn = KNeighborsRegressor(n_neighbors=k)
knn_w = KNeighborsRegressor(n_neighbors=k, weights='distance')
# Inspect the results.
y_pred = regr.predict(X_test)
print('\nmean-squared:', mean_squared_error(y_test, y_pred))
rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))
print('R^2 score: ', regr.score(X_test, y_test))
"""
Plan:
    mean-squared: 1338.6669779421761
    rms error is: 36.587798211182026
    KNN_dist R^2 score:  0.28078803137438835
"""
# Cross validate
score = cross_val_score(knn, X, y, cv=5, n_jobs=-1)
print("Unweighted R^2 score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score_w = cross_val_score(knn_w, X, y, cv=5, n_jobs=-1)
print("Weighted R^2 score: %0.2f (+/- %0.2f)" % (score_w.mean(), score_w.std() * 2))
#%%
# RandomForestRegressor:
# Random Forest: 
rfr = ensemble.RandomForestRegressor(n_jobs=-1, verbose=1)

# Choose some parameter combinations to try
parameters = {'n_estimators': [16, 32, 64, 96], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10, 13, 16], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 2, 5, 7]
             }

# Run the grid search
grid_obj = GridSearchCV(rfr, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
grid_obj.best_estimator_
#%%
# Run best model:
rfr = ensemble.RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=5,
           min_weight_fraction_leaf=0.0, n_estimators=16, n_jobs=-1,
           oob_score=False, random_state=None, verbose=1, warm_start=False)

rfr.fit(X_train, y_train) 
y_pred = rfr.predict(X_test)
print('\nmean-squared:', mean_squared_error(y_test, y_pred))
rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))
print('RandomForest R^2 score: ', rfr.score(X_test, y_test)) 
'''
Plain:
    mean-squared: 1349.1239181869814
    rms error is: 36.73042224351614
    RandomForest R^2 score:  0.275169937626511
    Cross Validated Score: 0.27 (+/- 0.08)
'''
# Cross validate
score = cross_val_score(rfr, X, y, cv=5, n_jobs=-1)
print("Cross Validated Score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%%
#SVM: 
svr = SVR(
        kernel='rbf', 
        degree=3, 
        gamma='scale', 
        coef0=0.0, tol=0.001, 
        C=1.0, 
        epsilon=0.1, 
        shrinking=True, 
        cache_size=200, 
        verbose=1, 
        max_iter=-1
        )

svr.fit(X_train, y_train) 
y_pred = svr.predict(X_test)
print('\nmean-squared:', mean_squared_error(y_test, y_pred))
rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))
print('SVM R^2 score: ', svr.score(X_test, y_test)) 
'''
Plain:
    mean-squared: 1494.0870372339377
    rms error is: 38.65342206369239
    SVM R^2 score:  0.1972870795701036
    Cross Validated Score: 0.20 (+/- 0.10)
'''
# Cross validate
score = cross_val_score(svr, X, y, cv=5, n_jobs=-1)
print("Cross Validated Score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%%
#Gradient Boosting
gbr = ensemble.GradientBoostingRegressor(n_estimators=500, n_iter_no_change=50, learning_rate=0.3)

# Choose some parameter combinations to try
parameters = {
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5, 7],
              'min_samples_leaf': [1, 2, 5, 7]
             }

# Run the grid search
grid_obj = GridSearchCV(gbr, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
grid_obj.best_estimator_
#%%
# Gradient Boosting: 
gbr = ensemble.GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.3, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=7,
             min_samples_split=5, min_weight_fraction_leaf=0.0,
             n_estimators=500, n_iter_no_change=50, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=1, warm_start=False)

gbr.fit(X_train, y_train) 
y_pred = gbr.predict(X_test)
print('\nmean-squared:', mean_squared_error(y_test, y_pred))
rmse_val = rmse(y_pred, y_test)
print("rms error is: " + str(rmse_val))
print('Gradient Boost R^2 score: ', gbr.score(X_test, y_test))
'''
Plain:
    mean-squared: 1240.92856103702
    rms error is: 35.22681593668413
    Gradient Boost R^2 score:  0.3332989548460096
    Cross Validated Score: 0.31 (+/- 0.07)
SelectKBest:
    mean-squared: 16920.14741134456
    rms error is: 130.0774669623629
    Gradient Boost R^2 score:  0.5508724897420324
    Cross Validated Score: -0.17 (+/- 0.64)
PCA:
    mean-squared: 1299.3694192763564
    rms error is: 36.0467671126879
    Gradient Boost R^2 score:  0.30190103034719595
    Cross Validated Score: 0.31 (+/- 0.08)
'''
# Cross validate
score = cross_val_score(gbr, X, y, cv=5, n_jobs=-1, verbose=1)
print("Cross Validated Score: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%% [markdown]
# #### Final model evaluation:
# The best model 

#### Other models
