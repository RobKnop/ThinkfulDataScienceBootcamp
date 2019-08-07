#%% [markdown]
# # Capstone Project Natural Language Processing (Unsupervised and Supervised Learning)
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass

import nltk
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 400)
pd.set_option('float_format', '{:.2f}'.format)
import re
import codecs
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter 
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
import gensim

# Load models
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn import metrics
#%% [markdown]
# ## Experiment Design
# Podcasts getting more and more mainstream in recent years. A lot of people are producing really good content, most of the time for free.  
# But what all this has to do with NLP? If you transcribe podcast you are converting the audio into a text respresentation. From there you are able to use the regularly NLP techniques. 
# In this experiment we wil use the transcript from the Tim Ferris Show: https://tim.blog/2018/09/20/all-transcripts-from-the-tim-ferriss-show/  
# ### Goal / Task
# 1. Cluster 10 different podcast guests or styles. See if you find appropriate clusters. Create some graphs to support the analysis.
# 2. Use Supervised modeling techniques to predict which podcast episode comes from which guest or style.  
#
# ### Data set
# 
# #### 10 different podcast guest or styles: 
# 
# 0. Tim Ferriss himself / solo (9x)
# 1. RadnomShow (8x)
# 2. Kevin Kelly (4x)
# 3. Dom D’Agostino (3x)
# 4. Tony Robins (4x)
# 5. Ramit Sethi (4x)
# 6. Waitzkin (4x)
# 7. Sacca (3x)
# 8. Peter Attias (3x)
# 9. Cal Fussman (3x)
# --> 45 episodes

#%%
# Import raw data
# This data set contains the transcripts of The Tim Ferriss Podcasts
# Source: https://tim.blog/2018/09/20/all-transcripts-from-the-tim-ferriss-show/
df = pd.read_csv("./data/tim-ferriss-podcast/thinkful.csv")
df.columns=['U', 'id', 'title', 'text']
df = df.drop(columns=['U'])
df.head()
#%%
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.replace(r"\\xa0", "")
    df[text_field] = df[text_field].str.replace(r"xa0", "")
    df[text_field] = df[text_field].str.replace(r" ''Tim Ferriss", "")
    df[text_field] = df[text_field].str.replace(r" n n''Tim Ferriss", "")
    df[text_field] = df[text_field].str.replace(r" n''Tim Ferriss", "")
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace(r"show notes and links at tim blog podcast tim ferriss", "")
    df[text_field] = df[text_field].str.replace(r"copyright   2007 2018 tim ferriss  all rights reserved", "")
    return df

df['title'] = df['title'].str.replace(r"The Tim Ferriss Show Transcripts: ", "")
df['title'] = df['title'].str.replace(r"Transcripts: ", "")
df['title'] = df['title'].str.replace(r"Tim Ferriss Show Transcript: ", "")
df['title'] = df['title'].str.replace(r"Episode ", "")

df = standardize_text(df, "text")
#%%
# Do standard NLP processing
# Tokenize
tokenizer = RegexpTokenizer(r'\w+')

df["tokens"] = df["text"].apply(tokenizer.tokenize)
df.head()

# StopWords
stop_words = set(stopwords.words('english'))
df["tokens"] = df["tokens"].map(lambda x: [w for w in x if not w in stop_words])

# Remove tokens tim and ferriss
df["tokens"] = df["tokens"].map(lambda x: [w for w in x if not w == 'tim'])
df["tokens"] = df["tokens"].map(lambda x: [w for w in x if not w == 'ferriss'])

# Lemmatize
lemmer = WordNetLemmatizer()
df["tokens"] = df["tokens"].map(lambda x: [lemmer.lemmatize(w) for w in x])

# Inspecting our dataset a little more

all_words = [word for tokens in df["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in df["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max episode length is %s" % max(sentence_lengths))
print("Mean episode length is %s" % np.mean(sentence_lengths))

# Define plotting functions

def plot_LSA(test_data, test_labels, plot=True):
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(test_data)
        lsa_scores = lsa.transform(test_data)
        if plot:
            x = lsa_scores[:,0]
            y = lsa_scores[:,1]
            plt.scatter(x, y, s=10, alpha=1)
            for i,label in enumerate(test_labels):
                plt.text(x[i], y[i], label, fontsize=14)

#%% [markdown]
# ## TF DF
#%%
# Define X and the date point label
X = df["text"].tolist()
label = df["title"].tolist()     

def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_tfidf, tfidf_vectorizer = tfidf(X)

# Plot TFIDF
fig = plt.figure(figsize=(50, 40))
plot_LSA(X_tfidf, label)
plt.show()

#%%
# Word2Vec
word2vec_path = "../../NLPofTimFerrissShow/pretrainedModels/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, df, generate_missing=False):
    embeddings = df['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                    generate_missing=generate_missing))
    return list(embeddings)

embeddings = get_word2vec_embeddings(word2vec, df)

# Plot Word2Vec
fig = plt.figure(figsize=(50, 40))         
plot_LSA(embeddings, label)
plt.show()

#%% [markdown]
# ## Clustering (Unsupervised)
#%%
df_emb = pd.DataFrame(embeddings)

# Calculate predicted values.
km = KMeans(n_clusters=10, random_state=42).fit(df_emb)
y_pred = km.predict(df_emb)

print('silhouette score: ', metrics.silhouette_score(df_emb, y_pred, metric='euclidean'))

# 2D
lsa = TruncatedSVD(n_components=2)
las_results = lsa.fit_transform(df_emb.values)
las_results = pd.DataFrame(las_results)

df_y = pd.DataFrame(y_pred, columns=['y_pred'])
df_y['y_pred'] = df_y['y_pred'].astype(int)

las_results = pd.concat([las_results, df_y], axis=1)

#Plot        
plt.scatter(las_results[0].values, las_results[1].values, c=y_pred)
plt.show()

# 3D
lsa = TruncatedSVD(n_components=3)
las_results = lsa.fit_transform(df_emb.values)
las_results = pd.DataFrame(las_results)

df_y = pd.DataFrame(y_pred, columns=['y_pred'])
df_y['y_pred'] = df_y['y_pred'].astype(int)

las_results = pd.concat([las_results, df_y], axis=1)

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(las_results[0].values, las_results[1].values, las_results[2].values, c=las_results['y_pred'])
pyplot.show()

# Combine df with y_pred to see cluster results
results = pd.concat([df, df_y], axis=1)
results = results.drop(columns=['text'])
#%%
# Visualize cluster result
# Plot the most frequent words per cluster
for i in range(0,9,1):
    counter = Counter([word for tokens in results[results["y_pred"] == i].tokens for word in tokens]) 
    pd.DataFrame(counter.most_common(12), columns=('most frequent words', 'count')).plot.bar('most frequent words',1, title='Cluster ' + str(i))
    plt.xticks(rotation=40)
    plt.show()
#%% [markdown]
# ### Findings from doing clustering:
# 
#%% [markdown]
# ## Supervised Learning Modeling
#%%
# Get the categories (y)
categories = pd.DataFrame(np.array([
    # Random Show
     [24,  1]
    ,[46,  1]
    ,[129, 1]
    ,[146, 1]
    ,[171, 1]
    ,[209, 1]
    ,[224, 1]
    ,[333, 1]
    # Kevin Kelly
    ,[27,  2]
    ,[96,  2]
    ,[164, 2]
    ,[247, 2]
    #Dom D’Agostino
    ,[117, 3]
    ,[172, 3]
    ,[188, 3]
    # Tony Robins 
    ,[37 , 4]
    ,[38 , 4]
    ,[178, 4]
    ,[186, 4]
    # Ramit Sethi
    ,[33,  5]
    ,[34,  5]
    ,[166, 5]
    ,[371, 5]
    # Waitzkin
    ,[2,   6]
    ,[148, 6]
    ,[204, 6]
    ,[375, 6]
    # Sacca
    ,[270, 7]
    ,[79 , 7]
    ,[132, 7]
    # Attia
    ,[50 , 8]
    ,[65 , 8]
    ,[352, 8]
    # Cal Fussman
    ,[145, 9]
    ,[183, 9]
    ,[259, 9]
    # Tim Ferriss Solo
    ,[319, 0]
    ,[49 , 0]
    ,[105, 0]
    ,[113, 0]
    ,[126, 0]
    ,[181, 0]
    ,[201, 0]
    ,[212, 0]
    ,[240, 0]
    ]),
    columns=['id', 'category'])
# Join with current dataframe "df"
df_model = pd.merge(df, categories, how='inner', on='id')

# Define X and y
X = df_model['text']
y = df_model['category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90)

#%%
# Use TFIDF
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Try Random Forrest
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=4, n_estimators=64)
# Fit the best algorithm to the data. 
rfc.fit(X_train_tfidf, y_train)
print('train: ', rfc.score(X_train_tfidf, y_train))
print('test: ', rfc.score(X_test_tfidf, y_test))
#%%
# Use word2vec embeddings
df_word2vec = pd.concat([df_emb, categories], axis=1)

# Define X and y
X = df_word2vec.drop(columns=['category'])
y = df_word2vec['category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(n_jobs=4)

# Choose some parameter combinations to try
parameters = {
                'n_estimators': [16, 32, 64, 96], 
                'max_features': ['log2', 'sqrt','auto'], 
                'criterion': ['entropy', 'gini'],
                'max_depth': [5, 10, 13], 
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
grid_obj.best_estimator_

#%%
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=13, max_features='log2', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=64, n_jobs=4,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)
print('train: ', rfc.score(X_train, y_train))
print('test: ', rfc.score(X_test, y_test))

y_pred = rfc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
score = cross_val_score(rfc, X, y, cv=5, n_jobs=-1, verbose=1)
print("RFC: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%% [markdown]
# ### Augment data

#%%
df_tmp = pd.DataFrame(columns=('id', 'title', 'text', 'category'))
split = 5
for index, row in df_model.iterrows():
    text = row.text
    size = round(len(text) / split)
    text_fragments = list(map(''.join, zip(*[iter(text)]*size)))
    for text in text_fragments:
        episode_dict = dict(
            {
            'id': row.id, 
            'title': row.title,
            'text': text, 
            'category': row.category 
            }
        )
        df_tmp = df_tmp.append(pd.DataFrame(
            episode_dict, index=[0]
            ))

# Define X and y
X = df_tmp['text']
y = df_tmp['category'].astype('int')

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#%%
# Use TFIDF
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Random Forest: 
rfc = ensemble.RandomForestClassifier(n_jobs=4)

# Choose some parameter combinations to try
parameters = {
                'n_estimators': [16, 32, 64, 96], 
                'max_features': ['log2', 'sqrt','auto'], 
                'criterion': ['entropy', 'gini'],
                'max_depth': [5, 10, 13], 
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X_train_tfidf, y_train)

# Set the clf to the best combination of parameters
grid_obj.best_estimator_

#%%
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=13, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=2, min_samples_split=5,
                       min_weight_fraction_leaf=0.0, n_estimators=96, n_jobs=4,
                       oob_score=False, random_state=None, verbose=0,
                       warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train_tfidf, y_train)
print('train: ', rfc.score(X_train_tfidf, y_train))
print('test: ', rfc.score(X_test_tfidf, y_test))

y_pred = rfc.predict(X_test_tfidf)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))

#%%
# Tokenize
tokenizer = RegexpTokenizer(r'\w+')

df_tmp["tokens"] = df_tmp["text"].apply(tokenizer.tokenize)
df_tmp.head()

# StopWords
stop_words = set(stopwords.words('english'))
df_tmp["tokens"] = df_tmp["tokens"].map(lambda x: [w for w in x if not w in stop_words])

# Remove tokens tim and ferriss
df_tmp["tokens"] = df_tmp["tokens"].map(lambda x: [w for w in x if not w == 'tim'])
df_tmp["tokens"] = df_tmp["tokens"].map(lambda x: [w for w in x if not w == 'ferriss'])

# Lemmatize
lemmer = WordNetLemmatizer()
df_tmp["tokens"] = df_tmp["tokens"].map(lambda x: [lemmer.lemmatize(w) for w in x])

#%%
embeddings = get_word2vec_embeddings(word2vec, df_tmp)
df_emb = pd.DataFrame(embeddings)

# Use word2vec embeddings
df_emb.reset_index(drop=True, inplace=True)
df_tmp.reset_index(drop=True, inplace=True)
df_word2vec = pd.concat([df_emb, df_tmp['category']], axis=1)

# Define X and y
X = df_word2vec.drop(columns=['category'])
y = df_word2vec['category'].astype(int)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(n_jobs=4, random_state=22)

# Choose some parameter combinations to try
parameters = {
                'n_estimators': [16, 32, 64, 96], 
                'max_features': ['log2', 'sqrt','auto'], 
                'criterion': ['entropy', 'gini'],
                'max_depth': [5, 6, 10, 13], 
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
grid_obj.best_estimator_

#%%
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
max_depth=6, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=5,
min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=4,
oob_score=False, random_state=22, verbose=0,
warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)
print('train: ', rfc.score(X_train, y_train))
print('test: ', rfc.score(X_test, y_test))

y_pred = rfc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
score = cross_val_score(rfc, X, y, cv=5, n_jobs=-1, verbose=1)
print("RFC: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))

#%%
for k in range(5, 25, 1):
    print('k = ', k)
    neighbors = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, weights='distance')
    neighbors.fit(X_train, y_train)

    print('train: ', neighbors.score(X_train, y_train))
    print('test: ', neighbors.score(X_test, y_test))
#%%
neighbors = KNeighborsClassifier(n_neighbors=8, n_jobs=-1, weights='distance')
neighbors.fit(X_train, y_train)

print('train: ', neighbors.score(X_train, y_train))
print('test: ', neighbors.score(X_test, y_test))

y_pred = neighbors.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('gbc:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
score = cross_val_score(neighbors, X, y, cv=5, n_jobs=-1, verbose=1)
print("gbc: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))
#%% [markdown]
# ### Findings:

#%% [markdown]
# ### Conclusion:
