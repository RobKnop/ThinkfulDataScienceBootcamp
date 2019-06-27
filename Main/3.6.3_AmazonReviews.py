#%% [markdown]
# # Amazon Reviews
# Use one of the following datasets to perform sentiment analysis on the given Amazon reviews. Pick one of the "small" datasets that is a reasonable size for your computer. The goal is to create a model to algorithmically predict if a review is positive or negative just based on its text. Try to see how these reviews compare across categories. Does a review classification model for one category work for another?
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


#%%
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# Read data
df = getDF('Main/data/amazon-reviews/reviews_Home_and_Kitchen_5.json.gz')
#%% [markdown]
# ### Variable descriptions
# * reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# * asin - ID of the product, e.g. 0000013714
# * reviewerName - name of the reviewer
# * helpful - helpfulness rating of the review, e.g. 2/3
# * reviewText - text of the review
# * overall - rating of the product
# * summary - summary of the review
# * unixReviewTime - time of the review (unix time)
# * reviewTime - time of the review (raw)

#%%
# Do first data profile report on raw data
pp.ProfileReport(df, check_correlation=False, pool_size=15).to_file(outputfile="3.6.3_AmazonReviews_RAW.html")
# https://github.com/RobKnop/ThinkfulDataScienceBootcamp/blob/master/Main/3.6.3_AmazonReviews_RAW.html
#%%
# Drop unnecessary columns
df = df.drop(columns=[
    'reviewerID',
    'asin', # 
    'reviewerName', # 
    'helpful', # 
    'unixReviewTime', 
    'reviewTime' # 
])
# Define the Y
df['y_sentiment'] = np.where(df['overall'] >= 4.0 , 1, 0)
df = df.drop(columns=['overall'])
# Drop duplicated
df = df.drop_duplicates() # Dataset has 70 duplicate rows
#%%
# Do second data profile report on cleaned data
pp.ProfileReport(df, check_correlation=False, pool_size=15).to_file(outputfile="3.6.3_AmazonReviews_CLEAN.html")
# See the webpage at: https://github.com/RobKnop/ThinkfulDataScienceBootcamp/blob/master/Main/3.6.3_AmazonReviews_CLEAN_5mio.html
#%%
# Feature Engineering
df['space'] = ' '
df['corpus'] = df['summary'] + df['space'] + df['reviewText']
df = df.drop(columns=[
    'summary',
    'reviewText',
    'space'
])
#%% [markdown]
# #### Findings
# 1. Correlation to y (delayed) exists: 
# 2. Multicollinearity is in general low, but certain variables are highly correlated
#   * like DEP_xxxx vars
#   * DISTANCE - DISTANCE_GROUP - AIR_TIME
# 3. Class imbalance: 17601697 - 2398303 (88%/12%)
#%% [markdown]
# #### Our key evaluation metric to optimize on is f1 score 
# * A balance between precision and recall is needed. 
#%% [markdown]
# #### Models to try:
# 1. LogisticRegression
# 2. Descion Tree 
# 3. Naive Bayes 
# 4. RandomForestClassifier
# 5. KNN
# 6. Support Vector Machine
# 7. GradientBoostingClassifier
# 8. (Also use of KSelectBest, GridSearch)
#%%
#Class Balancing via Under-Sampling
count_class_0, count_class_1 = df.y_sentiment.value_counts()

# Divide by class
df_class_0 = df[df['y_sentiment'] == 1]
df_class_1 = df[df['y_sentiment'] == 0]
print('Random under-sampling:')
df_class_0_under = df_class_0.sample(count_class_1)
df = pd.concat([df_class_0_under, df_class_1], axis=0)
print(df.y_sentiment.value_counts())

#%%
# Define X and y
X = TfidfVectorizer().fit_transform(df.corpus)
y = df['y_sentiment']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
# Logistic Regression: 
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50, verbose=1, n_jobs=2)

# Fit the model.
fit = lr.fit(X_train, y_train)

# Display.
y_pred = fit.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('LG:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('\nAUC: ', auc(fpr, tpr))
"""
Without Under-Sampling:
               precision    recall  f1-score   support

Under-Sampling:
           0       0.87      0.88      0.87     19242
           1       0.88      0.87      0.87     19348
"""
score = cross_val_score(fit, X, y, cv=5, scoring='f1', n_jobs=2)
print('\f1: ', score)
print("Cross Validated f1: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
# Cross Validated f1: 0.87 (+/- 0.00)
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
grid_obj = GridSearchCV(dt, parameters, scoring='recall', cv=3, n_jobs=2, verbose=1)
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
Without Under-Sampling:
               precision    recall  f1-score   support

Under-Sampling:
           0       0.50      1.00      0.67     19242
           1       0.00      0.00      0.00     19348
"""
score = cross_val_score(dt, X, y, cv=10, scoring='f1', n_jobs=2, verbose=1)
print("DT: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# DT: Input X --> f1: 0.810 (+/- 0.029)
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
Without Under-Sampling:
               precision    recall  f1-score   support

Under-Sampling:
           0       0.75      0.58      0.65     19242
           1       0.66      0.81      0.72     19348
"""
score = cross_val_score(bnb, X, y, cv=10, scoring='f1', n_jobs=2, verbose=1)
print("BNB: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
#BNB: Input X --> f1: 0.810 (+/- 0.005)
#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=2)

# Choose some parameter combinations to try
parameters = {'n_estimators': [16, 32, 64], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring='f1', cv=3, n_jobs=2, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

#%%
# Run best model:
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=13, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=32, n_jobs=2,
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
Without Under-Sampling:
              precision    recall  f1-score   support

Under-Sampling:
           0       0.73      0.70      0.71     19242
           1       0.71      0.74      0.72     19348
'''
score = cross_val_score(rfc, X, y, cv=10, scoring='f1', n_jobs=2, verbose=1)
print("RFC: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# RFC: Input X --> f1: 0.738 (+/- 0.010)
#%%
# KNN:
for k in range(6, 10, 1):
    print('k = ', k)
    neighbors = KNeighborsClassifier(n_neighbors=k, n_jobs=2, weights='distance')
    neighbors.fit(X_train, y_train)
    y_pred = neighbors.predict(X_test)
    #print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('KNN:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    #print('AUC: ', auc(fpr, tpr))
    # Cross Validation
    #score = cross_val_score(neighbors, X_test, y_test, cv=5, scoring='recall', n_jobs=2)
    #print("KNN: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
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
score = cross_val_score(svc, X_train, y_train, cv=5, scoring='f1', n_jobs=2, verbose=1)
print("Input X_train --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# Input X_train --> f1: 0.741 (+/- 0.014)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)
# Gradient Boosting
# We'll make 100 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 100,
          'max_depth': 2,
          'loss': 'deviance',
          'verbose': 1,
          'n_iter_no_change': 50, 
          'validation_fraction': 0.1,
          'learning_rate': 0.5
          }

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
Without Under-Sampling:
               precision    recall  f1-score   support
Under-Sampling
           0       0.95      0.95      0.95     24938
           1       0.95      0.95      0.95     25041
'''
score = cross_val_score(gbc, X, y, cv=10, scoring='f1', n_jobs=2, verbose=1)
print("GradBoost: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
"""
GradBoost: Input X --> f1: 0.987 (+/- 0.004) - elapsed: 12.4min
"""

#%% [markdown]
# #### Final model evaluation:
# The best model in all approaches is gradient boosting.
# Approaches are:
# * Without Under-Sampling --> 0.99/0.91 recall
# * Under-Sampling --> 0.95/0.95 recall
# * SelectKBest=100 --> 0.97/0.99 recall
# Every new approach boost the model performance, with SelectKBest winning in this case. 
# Other models could not cope with under-sampling or SelectKBest strategies as shown with gradient boosting.
# Looks like boosting really helps to learn the more difficult cases as well. 
# And that with no overfitting. The variance of the cross validation score is tiny.

#### Other models
# * SVM cannot handle that much data. The performance is therefore quite poor.  
# * KNN has a similar problem. It gets harder for KNN to process a lot of data.
# * RandomForest is the second best model. Specially with SelectKBest --> 0.95/0.94 recall
# * Decision tree could not improve much with under-sampling and got worse with SelectKBest
# * Naive Bayes is really fast to compute and results are quite good: SelectKBest --> 0.91/0.89 recall
# * Logistic Regression get very inefficient with a lot of data. The results are very good: Under-sampling --> 0.95/0.94 recall