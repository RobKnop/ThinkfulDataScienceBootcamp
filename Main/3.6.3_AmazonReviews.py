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
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD 
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
# filepath = 'Main/data/amazon-reviews/reviews_Home_and_Kitchen_5.json.gz'
filepath = 'reviews_Home_and_Kitchen_5.json.gz'
df = getDF(filepath)
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
# pp.ProfileReport(df, check_correlation=False, pool_size=15).to_file(outputfile="3.6.3_AmazonReviews_RAW.html")
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
# pp.ProfileReport(df, check_correlation=False, pool_size=15).to_file(outputfile="3.6.3_AmazonReviews_CLEAN.html")
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
# #### Our key evaluation metric to optimize on is accuracy, followed by the f1 score 
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

# Define X and y
X = TfidfVectorizer().fit_transform(df.corpus)
y = df['y_sentiment']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
# Logistic Regression: 
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50, verbose=1, n_jobs=-1)

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
score = cross_val_score(fit, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print('\f1: ', score)
print("Cross Validated f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# Cross Validated f1: 0.87 (+/- 0.01)
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
grid_obj = GridSearchCV(dt, parameters, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)
dt = grid_obj.best_estimator_
dt
#%%
#Run best DT model:

dt = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=13,
            max_features=1, max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=5,
            min_samples_split=3, min_weight_fraction_leaf=0.0,
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
score = cross_val_score(dt, X, y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
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
score = cross_val_score(bnb, X, y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("BNB: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
#BNB: Input X --> f1: 0.810 (+/- 0.005)
#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=-1)

# Choose some parameter combinations to try
parameters = {'n_estimators': [16, 32, 64], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring='accuracy', cv=3, n_jobs=10, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_
rfc
#%%
# Run best model:
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=13, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=64, n_jobs=-1,
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
           0       0.75      0.75      0.75     19242
           1       0.75      0.75      0.75     19348
'''
score = cross_val_score(rfc, X, y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("RFC: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# RFC: Input X --> f1: 0.739 (+/- 0.024)
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
           0       0.83      0.85      0.84      9682
           1       0.84      0.83      0.83      9613
'''
score = cross_val_score(gbc, X, y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("GradBoost: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
"""
GradBoost: Input X --> f1: 0.987 (+/- 0.004) - elapsed: 12.4min
"""
#%%
# Reduce dims
sklearn_tSVD = TruncatedSVD(n_components=5)
X_tSVD = sklearn_tSVD.fit_transform(X)
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tSVD, y, test_size=0.2, random_state=20)
#%%
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
# SVM:
svc = SVC(gamma='scale', verbose=1)
y_test
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('SVC:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
print('AUC: ', auc(fpr, tpr))
'''
               precision    recall  f1-score   support
           0       0.69      0.82      0.75     19242
           1       0.78      0.63      0.70     19348
'''
score = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
print("Input X_train --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
# Input X_train --> f1: 0.727 (+/- 0.005)
# %%
# KNN:
for k in range(5, 25, 1):
    print('k = ', k)
    neighbors = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    neighbors.fit(X_train, y_train)
    y_pred = neighbors.predict(X_test)
    #print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('KNN:\n', classification_report(y_test, y_pred, target_names=['0', '1']))
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
    #print('AUC: ', auc(fpr, tpr))
    # Cross Validation
    #score = cross_val_score(neighbors, X_test, y_test, cv=5, scoring='accuracy', n_jobs=-1)
    #print("KNN: Input X --> f1: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
'''
Without Under-Sampling:
               precision    recall  f1-score   support
k=19
Under-Sampling
           0       0.70      0.77      0.74     19242
           1       0.75      0.67      0.71     19348
'''
#%% [markdown]
# #### Final model evaluation:
# The best model logistic regression with a f1-score of 0.87.

#### Other models
# * SVM cannot has decent results and is not fast to compute. 
# * KNN has a similar problem. It gets harder for KNN to process a lot of data points.
# * A single Decision tree is not working because vectorized data.
# * RandomForest is better but with decent results. 
# * Naive Bayes is really fast to compute but the results are bad.
# * Gradient Boosting the second best model.