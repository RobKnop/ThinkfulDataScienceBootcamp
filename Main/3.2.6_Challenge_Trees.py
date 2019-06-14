#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Challenge: If a tree falls in the forest...
# Now that you've learned about random forests and decision trees let's do an exercise in accuracy. You know that random forests are basically a collection of decision trees. But how do the accuracies of the two models compare?
# 
# So here's what you should do. Pick a dataset. It could be one you've worked with before or it could be a new one. Then build the best decision tree you can.
# 
# Now try to match that with the simplest random forest you can. For our purposes measure simplicity with runtime. Compare that to the runtime of the decision tree. This is imperfect but just go with it.
# 
# Hopefully out of this you'll see the power of random forests, but also their potential costs. Remember, in the real world you won't necessarily be dealing with thousands of rows. It could be millions, billions, or even more.

#%%
import warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
#simplefilter(action='ignore', category=UndefinedMetricWarning)
import pandas as pd
pd.set_option('float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', 999)
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore", category=mpl.cbook.MatplotlibDeprecationWarning)
import seaborn as sns
from sklearn import ensemble, tree
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


#%%
df = pd.read_csv(
    "../Datasets/processed.cleveland.data",
    header=None,
    na_values='?'
)
df = df.dropna()# only 6 rows
df.columns = [
'age' ,    
'sex'     ,
'cp'   ,
'trestbps',
'chol'    ,
'fbs'    ,
'restecg' ,
'thalach',
'exang'  ,
'oldpeak',
'slope'  ,
'ca'    ,
'thal'  ,
'num' 
]
#df.thal.astype('float').astype('int')
pp.ProfileReport(df, check_correlation=True)

#%% [markdown]
# ## Correlation

#%%
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

sns.heatmap(df.corr(), vmax=.8, square=True)
plt.show()
print("Top Absolute Correlations")
print(get_top_abs_correlations(df, 30).to_string())


#%%
X = df.drop(columns=['num'])
y = df.num

#%% [markdown]
# ## Full features
# ### without Hyperparameter Tuning

#%%
rfc = ensemble.RandomForestClassifier()
rfc.fit(X, y)

dt = tree.DecisionTreeClassifier(
    criterion='entropy',
    max_features=1,
    max_depth=4,
    #random_state = 1338
)
dt.fit(X, y)

score = cross_val_score(rfc, X, y, cv=10)
print("RFC: Input X --> Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
score = cross_val_score(dt, X, y, cv=10)
print("DT: Input X --> Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
## RandomForest
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
y_pred = rfc.predict(X)
print('RFC:\n', classification_report(y, y_pred, target_names=target_names))
print('Confusion Matrix\n', confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4]))
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
print('AUC: ', auc(fpr, tpr))
# DecisionTree
y_pred = dt.predict(X)
print('\nDT:\n', classification_report(y, y_pred, target_names=target_names))
print('Confusion Matrix\n', confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4]))
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
print('AUC: ', auc(fpr, tpr))

#%% [markdown]
# #### Result: Random Forest performs much better than the Decision Tree
#%% [markdown]
# ## Hyperparameter Tuning
# 
# ### Random Forest Classifier:

#%%
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

rfc = ensemble.RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': range(1,50,5), 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, scoring=acc_scorer)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
rfc.fit(X, y)


#%%
score = cross_val_score(rfc, X, y, cv=10)
print("RFC: Input X --> Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
y_pred = rfc.predict(X)
print('RFC:\n', classification_report(y, y_pred, target_names=target_names))
print('Confusion Matrix\n', confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4]))
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
print('AUC: ', auc(fpr, tpr))

#%% [markdown]
# ### Decision Tree

#%%
parameters={'max_features' : range(1,10,1),'max_depth': range(1,10,1)}
clf_tree=tree.DecisionTreeClassifier()
clf=GridSearchCV(clf_tree,parameters)
clf.fit(X,y)


#%%
score = cross_val_score(clf, X, y, cv=10)
print("DT: Input X --> Accuracy: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
y_pred = clf.predict(X)
print('DT:\n', classification_report(y, y_pred, target_names=target_names))
print('Confusion Matrix\n', confusion_matrix(y, y_pred, labels=[0, 1, 2, 3, 4]))
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=2)
print('AUC: ', auc(fpr, tpr))

#%% [markdown]
# ## Selected Features 

#%%
rfc = ensemble.RandomForestClassifier()
for i in range(1,14):
    X_selKBest = SelectKBest(k=i).fit_transform(X, y)
    score = cross_val_score(rfc, X_selKBest, y, cv=10)
    print("k=%s, RFC: Input X_selKBest --> Accuracy: %0.3f (+/- %0.3f)" % (i, score.mean(), score.std() * 2))


#%%
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
# Instantiate and fit our model.
rfc = ensemble.RandomForestClassifier()

X_nor = StandardScaler().fit_transform(X)

for i in range(1,14):
    sklearn_pca = PCA(n_components=i)
    X_pca = sklearn_pca.fit_transform(X_nor)
    score = cross_val_score(rfc, X_pca, y, cv=10)
    print("components count=%s, RFC: Input X_selKBest --> Accuracy: %0.3f (+/- %0.3f)" % (i, score.mean(), score.std() * 2))

#%% [markdown]
# Best Pick: components count=10, RFC: Input X_selKBest --> Accuracy: 0.582 (+/- 0.084)

