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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

STORAGEACCOUNTNAME= os.environ.get('san')
STORAGEACCOUNTKEY= os.environ.get('sak')
CONTAINERNAME= os.environ.get('contname')
BLOBNAME= 'AirOnTime/2014_2012_AirOnTime.csv/AirOnTIme2004-2012.csv'
LOCALFILENAME= '2004_2012_AirOnTime.csv'
#%%
# Data Source: https://packages.revolutionanalytics.com/datasets/AirOnTime87to12/
# Data Description: https://packages.revolutionanalytics.com/datasets/AirOnTime87to12/AirOnTime87to12.dataset.description.txt

from azure.storage.blob import BlockBlobService
import time

#download from blob
t1=time.time()
blob_service=BlockBlobService(account_name=STORAGEACCOUNTNAME,account_key=STORAGEACCOUNTKEY)
blob_service.get_blob_to_path(CONTAINERNAME,BLOBNAME,LOCALFILENAME)
t2=time.time()
print(("It takes %s seconds to download "+BLOBNAME) % (t2 - t1))
#%% [markdown]
# Variable descriptions
YEAR: newName = "Year", type = "integer",
MONTH: newName = "Month", type = "integer",
DAY_OF_MONTH: newName = "DayofMonth", type = "integer",
DAY_OF_WEEK: newName = "DayOfWeek", type = "factor", 
    levels = as.character(1:7,
    newLevels = "Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun", 
FL_DATE: newName = "FlightDate", type = "character",
UNIQUE_CARRIER: newName = "UniqueCarrier", type = "factor",
TAIL_NUM: newName = "TailNum", type = "factor", aircraft registration code
FL_NUM: newName = "FlightNum", type = "factor",	
ORIGIN_AIRPORT_ID: newName = "OriginAirportID", type = "factor",
ORIGIN: newName = "Origin", type = "factor",
ORIGIN_CITY_NAME: newName = "OriginCityName", type = "factor",	
ORIGIN_STATE_ABR: newName = "OriginState", type = "factor",	
DEST_AIRPORT_ID: newName = "DestAirportID", type = "factor",
DEST: newName = "Dest", type = "factor",
DEST_CITY_NAME: newName = "DestCityName", type = "factor",	
DEST_STATE_ABR: newName = "DestState", type = "factor",		
CRS_DEP_TIME: newName = "CRSDepTime", type = "integer", scheduled local departure time
DEP_TIME: newName = "DepTime", type = "integer", 	actual departure time
DEP_DELAY: newName = "DepDelay", type = "integer", departure delay (includes negative delays: flights taking off before scheduled time), in minutes
DEP_DELAY_NEW: newName = "DepDelayMinutes", type = "integer", only positive departure delays, in minutes
DEP_DEL15: newName = "DepDel15", type = "logical",
DEP_DELAY_GROUP: newName = "DepDelayGroups", type = "factor",
   levels = as.character(-2:12),
   newLevels = "< -15", "-15 to -1","0 to 14", "15 to 29", "30 to 44",
    "45 to 59", "60 to 74", "75 to 89", "90 to 104", "105 to 119",
    "120 to 134", "135 to 149", "150 to 164", "165 to 179", ">= 180",
TAXI_OUT: newName = "TaxiOut", type =  "integer", moving on the aerodrome surface prior to take off
WHEELS_OFF: newName = "WheelsOff", type =  "integer", aircraft starts flying	
WHEELS_ON: newName = "WheelsOn", type =  "integer", aircraft landed
TAXI_IN: newName = "TaxiIn", type =  "integer", moving on the aerodrome surface prior to parking
CRS_ARR_TIME: newName = "CRSArrTime", type = "integer",	scheduled arrival time
ARR_TIME: newName = "ArrTime", type = "integer", 	Actual arrival time
ARR_DELAY: newName = "ArrDelay", type = "integer", arrival delay (includes negative delays: flights arriving before scheduled time), in minutes
ARR_DELAY_NEW: newName = "ArrDelayMinutes", type = "integer", only positive arrival delays, in minutes
ARR_DEL15: newName = "ArrDel15", type = "logical",
ARR_DELAY_GROUP: newName = "ArrDelayGroups", type = "factor",
  levels = as.character(-2:12),
   newLevels = "< -15", "-15 to -1","0 to 14", "15 to 29", "30 to 44",
    "45 to 59", "60 to 74", "75 to 89", "90 to 104", "105 to 119",
    "120 to 134", "135 to 149", "150 to 164", "165 to 179", ">= 180",
CANCELLED: newName = "Cancelled", type = "logical",
CANCELLATION_CODE: newName = "CancellationCode", type = "factor", 
    levels = "NA","A","B","C","D",	
        newLevels = "NA", "Carrier", "Weather", "NAS", "Security",
DIVERTED: newName = "Diverted", type = "logical", 
CRS_ELAPSED_TIME: newName = "CRSElapsedTime", type = "integer", estimated elapse time
ACTUAL_ELAPSED_TIME: newName = "ActualElapsedTime", type = "integer", 
AIR_TIME: newName = "AirTime", type =  "integer",
FLIGHTS: newName = "Flights", type = "integer",
DISTANCE: newName = "Distance", type = "integer",
DISTANCE_GROUP: newName = "DistanceGroup", type = "factor",
 levels = as.character(1:11),
 newLevels = "< 250", "250-499", "500-749", "750-999",
     "1000-1249", "1250-1499", "1500-1749", "1750-1999",
     "2000-2249", "2250-2499", ">= 2500",
CARRIER_DELAY: newName = "CarrierDelay", type = "integer", 	delays due to carrier, in minutes
WEATHER_DELAY: newName = "WeatherDelay", type = "integer", delays due to weather, in minutes
NAS_DELAY: newName = "NASDelay", type = "integer", delays due to national air system, in minutes
SECURITY_DELAY: newName = "SecurityDelay", type = "integer", delays due to security, in minutes
LATE_AIRCRAFT_DELAY: newName = "LateAircraftDelay", type = "integer", delays due to late aircraft, in minutes
#%%
#LOCALFILE is the file path
df = pd.read_csv(
                LOCALFILENAME, 
                parse_dates=[4], 
                na_values=' ')
#%%
pp.ProfileReport(df.iloc[:10000000], check_correlation=False, pool_size=15).to_file(outputfile="AirlineOnTime_RAW.html")
#%%
df = df.drop(columns=[
    '_c44',
    'FL_DATE', # dates are hard to process in ML models 
    'ORIGIN_STATE_ABR', # needs one_hot_encoding -> too inefficient 
    'DEST_STATE_ABR', # needs one_hot_encoding -> too inefficient
    'NAS_DELAY', # too much sematic regarding ARR_DELAY (the Y)
    'SECURITY_DELAY', # too much sematic regarding ARR_DELAY (the Y)
    'LATE_AIRCRAFT_DELAY', # too much sematic regarding ARR_DELAY (the Y)
    'WEATHER_DELAY', # too much sematic regarding ARR_DELAY (the Y)
    'CARRIER_DELAY', # too much sematic regarding ARR_DELAY (the Y)
    'FLIGHTS', # only values of 1
    'ARR_TIME', # too much sematic regarding ARR_DELAY (the Y)
    'ARR_DELAY_GROUP', # too much sematic regarding ARR_DELAY (the Y)
    'ARR_DEL15', # too much sematic regarding ARR_DELAY (the Y) and boolean
    'ACTUAL_ELAPSED_TIME', # too much sematic regarding ARR_DELAY (the Y) and boolean
    'ARR_DELAY', # has negative numbers but we are only interested in flights with a delay of > 30min
    'CANCELLATION_CODE' # too many missing values and no contribution to Y (assumption)
])
df = df.dropna(subset=['DEP_DEL15'])
df['WHEELS_OFF'] = pd.to_numeric(df['WHEELS_OFF'], errors='coerce')
df = df.dropna(subset=['WHEELS_OFF'])
df.fillna(0)
# Define the Y
df['y_delayed'] = np.where(df['ARR_DELAY_NEW'] > 30.0 , 1, 0)
df = df.drop(columns=['ARR_DELAY_NEW'])
# Clean more NaN
df = df.dropna() # (144'734 rows)
# Sample for higher iteration
df = df.sample(400000, random_state=1232)
#%%
pp.ProfileReport(df.iloc[:20000000], check_correlation=False, pool_size=15).to_file(outputfile="AirlineOnTime_CLEAN.html")

#%%
# Correlation
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations")
print(get_top_abs_correlations(df.drop(columns=['DEST', 'ORIGIN', 'UNIQUE_CARRIER', 'TAIL_NUM']), 50).to_string())
#%%
# Plot a heatmap to see all correlations between vars
plt.figure(figsize = (15,15))
sns_plot = sns.heatmap(df.corr(), vmax=.8, square=True)
sns_plot.get_figure().savefig('heatmap.png', bbox_inches='tight', dpi=200) 
sns_plot.get_figure().show()
#%% [markdown]
# #### Findings
# 1. Correlation to y (delayed) exists: 
# 2. Multicollinearity is in general low, but certain variables are highly correlated
#   * like DEP_xxxx vars
#   * DISTANCE - DISTANCE_GROUP - AIR_TIME
# 3. Class imbalance: 17601697 - 2398303 (88%/12%)
#%% [markdown]
# #### Our key evaluation metric to optimize on is recall 
# * For delay detection is more important to capture false negatives than false positives 
# * It is ok to predict an instance as delayed but it is not, because customer will be happy to hear that they are not late
# * On the other hand, it is not good to label an instance as NOT "delayed", but it was. That is bad expectation management
#%% [markdown]
# #### Models to try:
# 1. LogisticRegression
# 2. Descion Tree 
# 3. Naive Bayes 
# 4. RandomForestClassifier
# 5. KNN
# 6. Support Vector Machine
# 7. GradientBoostingClassifier

# PCA 
# SELECT KBest
# Class Balancing 
#%%
#RESAMPLE
# Normalize
mm_scaler = MinMaxScaler()
df[['YEAR']] = mm_scaler.fit_transform(df[['YEAR']].values)
df[['MONTH']] = mm_scaler.fit_transform(df[['MONTH']].values)
df[['DAY_OF_MONTH']] = mm_scaler.fit_transform(df[['DAY_OF_MONTH']].values)
df[['DAY_OF_WEEK']] = mm_scaler.fit_transform(df[['DAY_OF_WEEK']].values)
df[['FL_NUM']] = mm_scaler.fit_transform(df[['FL_NUM']].values)
df[['ORIGIN_AIRPORT_ID']] = mm_scaler.fit_transform(df[['ORIGIN_AIRPORT_ID']].values)
df[['DEST_AIRPORT_ID']] = mm_scaler.fit_transform(df[['DEST_AIRPORT_ID']].values)
df[['CRS_DEP_TIME']] = mm_scaler.fit_transform(df[['CRS_DEP_TIME']].values)
df[['DEP_TIME']] = mm_scaler.fit_transform(df[['DEP_TIME']].values)
df[['DEP_DELAY']] = mm_scaler.fit_transform(df[['DEP_DELAY']].values)
df[['DEP_DELAY_NEW']] = mm_scaler.fit_transform(df[['DEP_DELAY_NEW']].values)
df[['DEP_DEL15']] = mm_scaler.fit_transform(df[['DEP_DEL15']].values)
df[['DEP_DELAY_GROUP']] = mm_scaler.fit_transform(df[['DEP_DELAY_GROUP']].values)
df[['TAXI_OUT']] = mm_scaler.fit_transform(df[['TAXI_OUT']].values)
df[['WHEELS_OFF']] = mm_scaler.fit_transform(df[['WHEELS_OFF']].values)
df[['WHEELS_ON']] = mm_scaler.fit_transform(df[['WHEELS_ON']].values)
df[['TAXI_IN']] = mm_scaler.fit_transform(df[['TAXI_IN']].values)
df[['CRS_ARR_TIME']] = mm_scaler.fit_transform(df[['CRS_ARR_TIME']].values)
df[['CANCELLED']] = mm_scaler.fit_transform(df[['CANCELLED']].values)
df[['DIVERTED']] = mm_scaler.fit_transform(df[['DIVERTED']].values)
df[['CRS_ELAPSED_TIME']] = mm_scaler.fit_transform(df[['CRS_ELAPSED_TIME']].values)
df[['AIR_TIME']] = mm_scaler.fit_transform(df[['AIR_TIME']].values)
df[['DISTANCE']] = mm_scaler.fit_transform(df[['DISTANCE']].values)
df[['DISTANCE_GROUP']] = mm_scaler.fit_transform(df[['DISTANCE_GROUP']].values)
df[['y_delayed']] = mm_scaler.fit_transform(df[['y_delayed']].values)
#%%
# Define X and y
X = df.drop(columns=['y_delayed',
                     'ORIGIN',
                     'DEST',
                     'TAIL_NUM',
                     'FL_NUM',
                     'UNIQUE_CARRIER'
                    ])
X = pd.concat([X, pd.get_dummies(df['DEST'])], axis=1)
X = pd.concat([X, pd.get_dummies(df['ORIGIN'])], axis=1)

y = df['y_delayed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
# Logistic Regression: 
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50, verbose=1, n_jobs=10)

# Fit the model.
fit = lr.fit(X_train, y_train)

# Display.
y_pred = fit.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('LG:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('\nAUC: ', auc(fpr, tpr))
"""
               precision    recall  f1-score   support
           0       0.97      0.99      0.98   3524444
           1       0.89      0.78      0.83    475556
"""
score = cross_val_score(fit, X, y, cv=5, scoring='recall', n_jobs=-1)
print('\nRecall: ', score)
print("Cross Validated Recall: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# Cross Validated Recall: 0.78 (+/- 0.02)
#%%
# Decision Tree:
dt = tree.DecisionTreeClassifier()
parameters = { 
              'max_features': [1, 2, 3], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 3, 5, 8]
             }
# Run the grid search
grid_obj = GridSearchCV(dt, parameters, scoring='recall', cv=3, n_jobs=15, verbose=1)
grid_obj.fit(X, y)
dt = grid_obj.best_estimator_
#%%
#Run best DT model:

dt = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=10,
            max_features=3, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=8,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')

# Fit the best algorithm to the data. 
dt.fit(X_train, y_train)
#%%
# Evaluate
y_pred = dt.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('DT:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
"""
               precision    recall  f1-score   support
           0       0.98      0.99      0.98     88009
           1       0.89      0.82      0.85     11991
"""
score = cross_val_score(dt, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("DT: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# DT: Input X --> Recall: 0.810 (+/- 0.029)
#%%
# Naive Bayes:
bnb = BernoulliNB()
# Fit our model to the data.
bnb.fit(X_train, y_train)

# Evaluate
y_pred = bnb.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('BNB:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
"""
               precision    recall  f1-score   support
           0       0.98      0.91      0.95   3524444
           1       0.57      0.89      0.70    475556
"""
score = cross_val_score(bnb, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("BNB: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
#BNB: Input X --> Recall: 0.888 (+/- 0.001)
#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=17)

# Choose some parameter combinations to try
parameters = {'n_estimators': [16, 32, 64], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring='recall', cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

#%%
# Run best model:
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=13, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=32, n_jobs=17,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)

#%%
y_pred = rfc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
'''
              precision    recall  f1-score   support
           0       0.98      0.99      0.99    176169
           1       0.95      0.85      0.90     23831
'''
score = cross_val_score(rfc, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("RFC: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%%
# KNN:
for k in range(6, 40, 1):
#KNN with k = 19: Input X --> Recall: 0.643 (+/- 0.009)
    neighbors = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    neighbors.fit(X_train, y_train)
    y_pred = neighbors.predict(X_test)
    print('k = ', k)
    #print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('KNN:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    #print('AUC: ', auc(fpr, tpr))
    # Cross Validation
    #score = cross_val_score(neighbors, X_test, y_test, cv=5, scoring='recall', n_jobs=-1)
    #print("KNN: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
'''
k=8
               precision    recall  f1-score   support
           0       0.96      0.98      0.97    176169
           1       0.83      0.67      0.74     23831
'''
#%%
# SVM:
svc = SVC(gamma='scale', verbose=1)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('SVC:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
'''
               precision    recall  f1-score   support
           0       0.97      0.99      0.98     17642
           1       0.90      0.76      0.82      2358
'''
score = cross_val_score(svc, X_train, y_train, cv=5, scoring='recall', n_jobs=-1, verbose=1)
print("Input X_train --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# Input X_train --> Recall: 0.741 (+/- 0.014)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
# Gradient Boosting
# We'll make 500 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 500,
          'max_depth': 2,
          'loss': 'deviance',
          'verbose': 1,
          'n_iter_no_change': 50, 
          'validation_fraction': 0.1}

# Initialize and fit the model.
gbc = ensemble.GradientBoostingClassifier(**params)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('GradBoost:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
# Best:
'''
               precision    recall  f1-score   support
           0       0.99      0.99      0.99     88009
           1       0.96      0.91      0.93     11991
'''
score = cross_val_score(gbc, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("GradBoost: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
"""
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 32 concurrent workers.
[Parallel(n_jobs=-1)]: Done   2 out of  10 | elapsed: 19.6min remaining: 78.4min
[Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed: 20.2min finished
GradBoost: Input X --> Recall: 0.906 (+/- 0.005)
"""

#%% [markdown]
# #### Final model evaluation:


#%%
