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
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
# Source: http://stat-computing.org/dataexpo/2009/the-data.html

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
#       Name	            Description
# 1	    Year	            1987-2008
# 2	    Month	            1-12
# 3	    DayofMonth	        1-31
# 4	    DayOfWeek	        1 (Monday) - 7 (Sunday)
# 5	    DepTime	            actual departure time (local, hhmm)
# 6	    CRSDepTime	        scheduled departure time (local, hhmm)
# 7	    ArrTime	            actual arrival time (local, hhmm)
# 8	    CRSArrTime	        scheduled arrival time (local, hhmm)
# 9	    UniqueCarrier	    unique carrier code
# 10	FlightNum	        flight number
# 11	TailNum	plane       tail number
# 12	ActualElapsedTime	in minutes
# 13	CRSElapsedTime	    in minutes
# 14	AirTime	            in minutes
# 15	ArrDelay	        arrival delay, in minutes
# 16	DepDelay	        departure delay, in minutes
# 17	Origin	            origin IATA airport code
# 18	Dest	            destination IATA airport code
# 19	Distance	        in miles
# 20	TaxiIn	            taxi in time, in minutes
# 21	TaxiOut	            taxi out time in minutes
# 22	Cancelled	        was the flight cancelled?
# 23	CancellationCode	reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
# 24	Diverted	        1 = yes, 0 = no
# 25	CarrierDelay	    in minutes
# 26	WeatherDelay	    in minutes
# 27	NASDelay	        in minutes
# 28	SecurityDelay	    in minutes
# 29	LateAircraftDelay	in minutes
#%%
#LOCALFILE is the file path
df = pd.read_csv(
                LOCALFILENAME, 
                parse_dates=[4], 
                #dtype={"UNIQUE_CARRIER": str}, 
                na_values=' ')

df = df.drop(columns=[
    '_c44',
    'FL_DATE', 
    'ORIGIN_STATE_ABR',
    'DEST_STATE_ABR', 
    'NAS_DELAY',
    'SECURITY_DELAY', 
    'LATE_AIRCRAFT_DELAY', 
    'WEATHER_DELAY', 
    'CARRIER_DELAY',
    'FLIGHTS' # only values of 1
    'ARR_DELAY_GROUP', # too much sematic regarding ARR_DELAY (the Y)
    'ARR_DEL15', # too much sematic regarding ARR_DELAY (the Y)
    'ARR_DELAY_NEW'
])
df = df.dropna(subset=['DEP_DEL15'])
df.fillna(0)
#%%
df['delayed'] = np.where(df['ARR_DELAY'] > 30.0 , 1, 0)
#%%
pp.ProfileReport(df, check_correlation=False, pool_size=1).to_file(outputfile="AirlineOnTime.html")
#%%
#df['WHEELS_OFF'] = np.where(df['WHEELS_OFF']str.contains('-') , NaN, df['WHEELS_OFF']str.astype('float'))
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
print(get_top_abs_correlations(df, 50).to_string())
#%%
plt.figure(figsize = (15,15))
sns_plot = sns.heatmap(df.corr(), vmax=.8, square=True)
sns_plot.get_figure().savefig('heatmap.png', bbox_inches='tight', dpi=200) 
#%% [markdown]
# #### Findings
# 1. Time, Amount and Class has some correlation with Vxx
# 2. Multicollinearity is low
# 3. High class imbalance
#%% [markdown]
# #### Our key evaluation metric to optimize on is recall 
# * For fraud prevention is more important to capture false negatives than false positives 
# * It is ok to predict an instance as fraud but it is not, because there is no direct money loss for the bank and customer
# * On the other hand, it is NOT ok to label an instance as NOT fraud, but it was. There is direct money loss for the customer  
#%% [markdown]
# #### Models to try:
# 1. LogisticRegression
# 2. RandomForestClassifier
# 3. KNN
# 4. Support Vector Machine
# 5. GradientBoostingClassifier
# 6. Descion Tree
# 7. Naive Bayes 

# PCA 
# SELECT KBest
# Class Balancing 
#%%
mm_scaler = MinMaxScaler()
df[['Time']] = mm_scaler.fit_transform(df[['Time']].values)
df[['Amount']] = mm_scaler.fit_transform(df[['Amount']].values)

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
# Logistic Regression: 
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)

# Fit the model.
fit = lr.fit(X_train, y_train)

# Display.
y_pred = fit.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('\nAUC: ', auc(fpr, tpr))
score = cross_val_score(fit, X, y, cv=5, scoring='recall')
print('\nRecall: ', score)
print("Cross Validated Recall: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 8, 16, 32, 64], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

''' 
Best Model so far:
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=13, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=64, n_jobs=-1,
            oob_score=False, random_state=None, verbose=1,
            warm_start=False)
'''

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring='recall', cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

#%%
# Run best model:
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=64, n_jobs=-1,
            oob_score=False, random_state=None, verbose=1,
            warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)

#%%
y_pred = rfc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))

#%%
score = cross_val_score(rfc, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("RFC: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%%
# KNN:
# for k in range(4, 40, 1):
k = 19
neighbors = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
neighbors.fit(X_train, y_train)
y_pred = neighbors.predict(X_test)
print('k = ', k)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('KNN:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
# Cross Validation
score = cross_val_score(neighbors, X_test, y_test, cv=5, scoring='recall', n_jobs=-1)
print("KNN: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
#%%
# SVM:
svc = SVC(gamma='scale')
score = cross_val_score(svc, X_train, y_train, cv=5, scoring='recall', n_jobs=-1, verbose=1)
print("Input X_train --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
## Recall: 0.000 (+/- 0.000) --> Not working
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

#predict_train = gbc.predict(X_train)
y_pred = gbc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
# Best:
#              precision    recall  f1-score   support
#           0       1.00      1.00      1.00     28314
#           1       0.91      0.69      0.79        59
#%%
score = cross_val_score(gbc, X, y, cv=10, scoring='recall', n_jobs=-1, verbose=1)
print("GradBoost: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# Output> Recall: 0.456 (+/- 0.691) --> too much variance --> unstable results 

#%% [markdown]
# #### Final model evaluation:


#%%