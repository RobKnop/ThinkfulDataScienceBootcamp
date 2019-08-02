#%%
import nltk
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 400)
pd.set_option('float_format', '{:.2f}'.format)
import re
import codecs
import matplotlib.pyplot as plt
import IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim

# Load models
from sklearn import ensemble, tre
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn import metrics
#%% 
#%% [markdown]
# 10 Guest and styles: 
# 
# 1. RadnomShow (5x)
# 2. Kevin Kelly (4x)
# 3. Dom D’Agostino (3x)
# 4. Tim Ferriss himself
# 5. Tony Robins
# 6. Ramit Sethi
# 7, Waitzkin
# 8. Sacca 
# 9. Peter Attias
# 10. Cal Fussman
#%%
# Import raw data
# This data set contains the transcripts of Tim Ferriss Podcasts
# Source: https://tim.blog/2018/09/20/all-transcripts-from-the-tim-ferriss-show/
df = pd.read_csv("./Main/data/tim-ferriss-podcast/transcripts.csv")
df.columns=['id', 'title', 'text', 'class']
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
    return df

df['title'] = df['title'].str.replace(r"The Tim Ferriss Show Transcripts: ", "")
df['title'] = df['title'].str.replace(r"Transcripts: ", "")
df = standardize_text(df, "text")

df.to_csv("./cleaned/transcripts_cleaned.csv")

#%%
clean_questions = pd.read_csv("./cleaned/transcripts_cleaned.csv")
clean_questions.head()

#%%
tokenizer = RegexpTokenizer(r'\w+')

clean_questions["tokens"] = clean_questions["text"].apply(tokenizer.tokenize)
clean_questions.head()

#  StopWords
stop_words = set(stopwords.words('english'))
clean_questions["tokens"] = clean_questions["tokens"].map(lambda x: [w for w in x if not w in stop_words])

# Lemmatize
# TODO

# Inspecting our dataset a little more

all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max episode length is %s" % max(sentence_lengths))
print("Mean episode length is %s" % np.mean(sentence_lengths))

def plot_TSNE(test_data, test_labels, plot=True):
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, early_exaggeration=20)
        tsne_results = tsne.fit_transform(test_data) #for TFIDF -> test_data.toarray()
        #print(tsne_results)
        if plot:
            x = tsne_results[:,0]
            y = tsne_results[:,1]
            plt.scatter(x, y, s=10, alpha=1)
            for i,label in enumerate(test_labels):
                plt.text(x[i], y[i], label, fontsize=14)

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
# # Bag of Words Counts 

#%%
def cv(data):
    count_vectorizer = CountVectorizer(stop_words='english')

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

X_train = clean_questions["text"].tolist()
y_train = clean_questions["title"].tolist()       

X_train_counts, count_vectorizer = cv(X_train)

# Plot Embeddings

fig = plt.figure(figsize=(160, 80))
plot_LSA(X_train_counts, y_train)
plt.show()

#%% [markdown]
# ## TF DF

#%%
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)


#%%

# Plot TFIDF

fig = plt.figure(figsize=(160, 80))
plot_LSA(X_train_tfidf, y_train)
plt.show()

fig = plt.figure(figsize=(160, 100))
plot_TSNE(X_train_tfidf.toarray(), y_train)
plt.show()

#%%
# Word2Vec
word2vec_path = "./pretrainedModels/GoogleNews-vectors-negative300.bin"
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

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

embeddings = get_word2vec_embeddings(word2vec, clean_questions)
#%%
fig = plt.figure(figsize=(160, 80))         
plot_LSA(embeddings, y_train)
plt.show()

fig = plt.figure(figsize=(160, 100))
plot_TSNE(embeddings, y_train)
plt.show()

#%%
# Modeling

## get the categories

categories = pd.read_csv("./cleaned/clean_categories.csv")
categories = categories.drop(columns=['id', 'title'])

df = pd.DataFrame()
df = pd.concat([clean_questions, categories], axis=1)
df = df.dropna()

X = df['text']

y = df['category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#%%
# Use TFIDF
X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=4)
# Fit the best algorithm to the data. 
rfc.fit(X_train_tfidf, y_train)
print('train: ', rfc.score(X_train_tfidf, y_train))
print('test: ', rfc.score(X_test_tfidf, y_test))

y_pred = rfc.predict(X_test_tfidf)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3']))
#%%
# Use word2vec embeddings
df = pd.DataFrame(embeddings)
df = pd.concat([df, categories], axis=1)
df = df.dropna()

X = df.drop(columns=['category', 'episode'])

y = df['category']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

#%%
# Random Forest: 
rfc = ensemble.RandomForestClassifier(criterion='entropy', n_jobs=4)

# Choose some parameter combinations to try
parameters = {'n_estimators': [16, 32, 64], 
              #'max_features': ['log2', 'sqrt','auto'], 
              #'criterion': ['entropy', 'gini'],
              'max_depth': [5, 10, 13], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 2, 5]
             }

# Run the grid search
grid_obj = GridSearchCV(rfc, parameters, cv=3, n_jobs=-1, verbose=1)
grid_obj.fit(X, y)

# Set the clf to the best combination of parameters
rfc = grid_obj.best_estimator_

#%%
rfc = ensemble.RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
            max_depth=5, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=32, n_jobs=4,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)

# Fit the best algorithm to the data. 
rfc.fit(X_train, y_train)
print('train: ', rfc.score(X_train, y_train))
print('test: ', rfc.score(X_test, y_test))

y_pred = rfc.predict(X_test)
print('Confusion Matrix\n', pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('RFC:\n', classification_report(y_test, y_pred, target_names=['0', '1', '2', '3']))
score = cross_val_score(rfc, X, y, cv=5, n_jobs=-1, verbose=1)
print("RFC: Input X --> Recall: %0.3f (+/- %0.3f)" % (score.mean(), score.std() * 2))


#%% [markdown]
# ### Findings: 
# Best model is the last rfc with word2vec embeddings:
# Increased RFC: Input X --> Recall: 0.534 (+/- 0.273) to 0.656 (+/- 0.114)

#%% 
## Clustering
df = pd.DataFrame(embeddings)

# Calculate predicted values.
km = KMeans(n_clusters=5, random_state=42).fit(df)
y_pred = km.predict(df)



tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, early_exaggeration=20)
tsne_results = tsne.fit_transform(df.values)
# Plot the solution.
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y_pred)
plt.show()

print('silhouette score', metrics.silhouette_score(df, y_pred, metric='euclidean'))

lsa = TruncatedSVD(n_components=2)
las_results = lsa.fit_transform(df.values)
las_results = pd.DataFrame(las_results)
df_y = pd.DataFrame(y_pred, columns=['y_pred'])

df_y['y_pred'] = df_y['y_pred'].astype(int)

las_results = pd.concat([las_results, df_y], axis=1)
#%%
fig = plt.figure(figsize=(80, 60))         
plt.scatter(las_results[0].values, las_results[1].values, c=y_pred)

for i, txt in enumerate(clean_questions['title']):
    plt.annotate(txt, (las_results[0].values[i], las_results[1].values[i]))
"""
texts = []
for x, y, s in zip(las_results[0].values, las_results[1].values, clean_questions['title']):
    texts.append(plt.text(x, y, s))
adjust_text(texts, only_move='y', arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
"""
plt.show()

#%% 
## Clustering
df = pd.DataFrame(embeddings)

# Calculate predicted values.
km = KMeans(n_clusters=5, random_state=42).fit(df)
y_pred = km.predict(df)



tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300, early_exaggeration=20)
tsne_results = tsne.fit_transform(df.values)
tsne_results = pd.DataFrame(tsne_results)
df_y = pd.DataFrame(y_pred, columns=['y_pred'])

df_y['y_pred'] = df_y['y_pred'].astype(int)

tsne_results = pd.concat([tsne_results, df_y], axis=1)
# Plot the solution.
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(tsne_results[0], tsne_results[1], tsne_results[2], c=tsne_results['y_pred'])
pyplot.show()


#%%

lsa = TruncatedSVD(n_components=3)
las_results = lsa.fit_transform(df.values)
las_results = pd.DataFrame(las_results)
df_y = pd.DataFrame(y_pred, columns=['y_pred'])

df_y['y_pred'] = df_y['y_pred'].astype(int)

las_results = pd.concat([las_results, df_y], axis=1)
# Plot the solution.
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(las_results[0].values, las_results[1].values, las_results[2].values, c=las_results['y_pred'])
pyplot.show()


#%%
results = pd.concat([clean_questions, df_y], axis=1)
results = results.drop(columns=['text', 'tokens'])