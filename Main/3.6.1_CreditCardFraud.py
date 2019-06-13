#%%
from IPython import get_ipython
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
import pandas_profiling as pp
import numpy as np
import scipy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, make_scorer, average_precision_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import normalize
#%%
df = pd.read_csv("./creditcard.csv") 
## new features
df.head()
#%%
pp.ProfileReport(df, check_correlation=True).to_file(outputfile="ProfileOfCCFraud.html")
#%%
# Drop duplicated?
df = df.drop_duplicates()
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
print(get_top_abs_correlations(df, 30).to_string())

sns.heatmap(df.corr(), vmax=.8, square=True)
plt.figure(figsize=(16, 16))
plt.show()
#%% [markdown]
# #### Findings
# 1. Time, Amount and Class has some correlation with Vxx
# 2. Multicollinearity is low
# 3. High class imbalance
#%%
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
#%%
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)

# Fit the model.
fit = lr.fit(X_train, y_train)

# Display.
y_pred = fit.predict(X_test)
print('Confusion Matrix\n', confusion_matrix(y_test, y_pred, labels=[0, 1]))
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print('\n Percentage accuracy')
print(fit.score(X_test, y_test))

score = cross_val_score(fit, X, y, cv=5, scoring='precision')
print('\nPrecision: ', score)
print("Cross Validated Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
score = cross_val_score(fit, X, y, cv=5, scoring='recall')
print('\nRecall: ', score)
print("Cross Validated Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))


#%%
rfc = ensemble.RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': range(1,30,5), 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              #'min_samples_split': [2, 3, 5],
              #'min_samples_leaf': [1,5,8]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring='precision')
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X, y)


#%%
#score = cross_val_score(rfc, X, y, cv=10)
#print("RFC: Input X --> Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
target_names = ['0', '1']
y_pred = rfc.predict(X)
print('RFC:\n', classification_report(y, y_pred, target_names=target_names))
print('Confusion Matrix\n', confusion_matrix(y, y_pred, labels=[0, 1]))
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
print('AUC: ', auc(fpr, tpr))