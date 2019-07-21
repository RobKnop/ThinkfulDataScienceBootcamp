#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'Main'))
	print(os.getcwd())
except:
	pass

#%%
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import scipy
import sklearn
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import gutenberg, stopwords
from collections import Counter, defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn import linear_model

nltk.download('gutenberg')

#%% [markdown]
# Supervised NLP requires a pre-labelled dataset for training and testing, and is generally interested in categorizing text in various ways. In this case, we are going to try to predict whether a sentence comes from _Alice in Wonderland_ by Lewis Carroll or _Persuasion_ by Jane Austen. We can use any of the supervised models we've covered previously, as long as they allow categorical outcomes. In this case, we'll try Random Forests, SVM, and KNN.
# 
# Our feature-generation approach will be something called _BoW_, or _Bag of Words_. BoW is quite simple: For each sentence, we count how many times each word appears. We will then use those counts as features.
# 
# **Note**: Since processing all the text takes around ~5-10 minutes, in the cell below we are taking only the first tenth of each text. If you want to experiment, feel free to change the following code in the next cell:
# 
# ```python
# alice = text_cleaner(alice[:int(len(alice)/10)])
# persuasion = text_cleaner(persuasion[:int(len(persuasion)/10)])
# ```
# to 
# 
# ```python
# alice = text_cleaner(alice)
# persuasion = text_cleaner(persuasion)
# ```

#%%
# Utility function for standard text cleaning.
def text_cleaner(text):
    # Visual inspection identifies a form of punctuation spaCy does not
    # recognize: the double dash '--'.  Better get rid of it now!
    text = re.sub(r'--',' ',text)
    text = re.sub("[\[].*?[\]]", "", text)
    text = ' '.join(text.split())
    return text
    
# Load and clean the data.
persuasion = gutenberg.raw('austen-persuasion.txt')
alice = gutenberg.raw('carroll-alice.txt')

# The Chapter indicator is idiosyncratic
persuasion = re.sub(r'Chapter \d+', '', persuasion)
alice = re.sub(r'CHAPTER .*', '', alice)
    
alice = text_cleaner(alice)
persuasion = text_cleaner(persuasion)


#%%
# Parse the cleaned novels. This can take a bit.
nlp = spacy.load('en')
alice_doc = nlp(alice)
persuasion_doc = nlp(persuasion)


#%%
# Group into sentences.
alice_sents = [[sent, "Carroll"] for sent in alice_doc.sents]
persuasion_sents = [[sent, "Austen"] for sent in persuasion_doc.sents]

# Combine the sentences from the two novels into one data frame.
sentences = pd.DataFrame(alice_sents + persuasion_sents)
sentences.head()

#%% [markdown]
# Time to bag some words!  Since spaCy has already tokenized and labelled our data, we can move directly to recording how often various words occur.  We will exclude stopwords and punctuation.  In addition, in an attempt to keep our feature space from exploding, we will work with lemmas (root words) rather than the raw text terms, and we'll only use the 2000 most common words for each text.

#%%
# Utility function to create a list of the 2000 most common words.
def bag_of_words(text):
    
    # Filter out punctuation and stop words.
    allwords = [token.lemma_
                for token in text
                if not token.is_punct
                and not token.is_stop]
    
    # Return the most common words.
    return [item[0] for item in Counter(allwords).most_common(2000)]
    

# Creates a data frame with features for each word in our common word set.
# Each value is the count of the times the word appears in each sentence.
def bow_features(sentences, common_words):
    
    # Scaffold the data frame and initialize counts to zero.
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:, common_words] = 0
    
    # Process each row, counting the occurrence of words in each sentence.
    for i, sentence in enumerate(df['text_sentence']):
        
        # Convert the sentence to lemmas, then filter out punctuation,
        # stop words, and uncommon words.
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words
                 )]
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i, word] += 1
        
        # This counter is just to make sure the kernel didn't hang.
        if i % 50 == 0:
            print("Processing row {}".format(i))
            
    return df

# Set up the bags.
alicewords = bag_of_words(alice_doc)
persuasionwords = bag_of_words(persuasion_doc)

# Combine bags to create a set of unique words.
common_words = set(alicewords + persuasionwords)


#%%
# Create our data frame with features. This can take a while to run.
word_counts = bow_features(sentences, common_words)
word_counts.head()

#%% [markdown]
# ## Trying out BoW
# 
# Now let's give the bag of words features a whirl by trying a random forest.

#%%
from sklearn import ensemble
from sklearn.model_selection import train_test_split

rfc = ensemble.RandomForestClassifier()
Y = word_counts['text_source']
X = np.array(word_counts.drop(['text_sentence','text_source'], 1))

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.4,
                                                    random_state=0)
train = rfc.fit(X_train, y_train)

print('Training set score:', rfc.score(X_train, y_train))
print('\nTest set score:', rfc.score(X_test, y_test))

#%% [markdown]
# Holy overfitting, Batman! Overfitting is a known problem when using bag of words, since it basically involves throwing a massive number of features at a model – some of those features (in this case, word frequencies) will capture noise in the training set. Since overfitting is also a known problem with Random Forests, the divergence between training score and test score is expected.
# 
# 
# ## BoW with Logistic Regression
# 
# Let's try a technique with some protection against overfitting due to extraneous features – logistic regression with ridge regularization (from ridge regression, also called L2 regularization).

#%%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # No need to specify l2 as it's the default. But we put it for demonstration.
train = lr.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', lr.score(X_train, y_train))
print('\nTest set score:', lr.score(X_test, y_test))
score = cross_val_score(lr, X, Y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("LR: Input X --> %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

#%% [markdown]
# Logistic regression performs a bit better than the random forest.  
# 
# # BoW with Gradient Boosting
# 
# And finally, let's see what gradient boosting can do:

#%%
clf = ensemble.GradientBoostingClassifier()
train = clf.fit(X_train, y_train)

print('Training set score:', clf.score(X_train, y_train))
print('\nTest set score:', clf.score(X_test, y_test))

#%% [markdown]
# Looks like logistic regression is the winner, but there's room for improvement.
# 
# # Same model, new inputs
# 
# What if we feed the model a different novel by Jane Austen, like _Emma_?  Will it be able to distinguish Austen from Carroll with the same level of accuracy if we insert a different sample of Austen's writing?
# 
# First, we need to process _Emma_ the same way we processed the other data, and combine it with the Alice data. Remember that for computation time concerns, we only took the first tenth of the Alice text. Emma is pretty long. **So in order to get comparable length texts, we take the first sixtieth of Emma**. Again, if you want to experiment, you can take the whole texts of each.

#%%
# Clean the Emma data.
emma = gutenberg.raw('austen-emma.txt')
emma = re.sub(r'VOLUME \w+', '', emma)
emma = re.sub(r'CHAPTER \w+', '', emma)
emma = text_cleaner(emma[:int(len(emma)/60)])
print(emma[:100])

# Parse our cleaned data.
emma_doc = nlp(emma)

# Group into sentences.
emma_sents = [[sent, "Austen"] for sent in emma_doc.sents]

# Build a new Bag of Words data frame for Emma word counts.
# We'll use the same common words from Alice and Persuasion.
emma_sentences = pd.DataFrame(emma_sents)
emma_bow = bow_features(emma_sentences, common_words)

print('done')


#%%
# Now we can model it!
# Let's use logistic regression again.

# Combine the Emma sentence data with the Alice data from the test set.
X_Emma_test = np.concatenate((
    X_train[y_train[y_train=='Carroll'].index],
    emma_bow.drop(['text_sentence','text_source'], 1)
), axis=0)
y_Emma_test = pd.concat([y_train[y_train=='Carroll'],
                         pd.Series(['Austen'] * emma_bow.shape[0])])

# Model.
print('\nTest set score:', lr.score(X_Emma_test, y_Emma_test))
lr_Emma_predicted = lr.predict(X_Emma_test)
pd.crosstab(y_Emma_test, lr_Emma_predicted)

#%% [markdown]
# Well look at that!  NLP approaches are generally effective on the same type of material as they were trained on. It looks like this model is actually able to differentiate multiple works by Austen from Alice in Wonderland.  Now the question is whether the model is very good at identifying Austen, or very good at identifying Alice in Wonderland, or both...
# 
# # Challenge 0:
# 
# Recall that the logistic regression model's best performance on the test set was 93%.  See what you can do to improve performance.  Suggested avenues of investigation include: Other modeling techniques (SVM?), making more features that take advantage of the spaCy information (include grammar, phrases, POS, etc), making sentence-level features (number of words, amount of punctuation), or including contextual information (length of previous and next sentences, words repeated from one sentence to the next, etc), and anything else your heart desires.  Make sure to design your models on the test set, or use cross_validation with multiple folds, and see if you can get accuracy above 90%.  
#%% [markdown]
# ### Other model:

#%%
scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)
# SVM:
svc = SVC(gamma='scale')

svc.fit(X_train, y_train)
print('Training set score:', svc.score(X_train, y_train))
print('\nTest set score:', svc.score(X_test, y_test))

#%% [markdown]
# ### making more features that take advantage of the spaCy information (include grammar, phrases, POS, etc
#%%
pos_counts = defaultdict(Counter)
df_pos = pd.DataFrame(columns=['POS', 'count', 'word'])

for token in alice_doc:
    pos_counts[token.pos][token.orth] += 1

for pos_id, counts in sorted(pos_counts.items()):
    pos = alice_doc.vocab.strings[pos_id]
    for orth_id, count in counts.most_common():
        df_pos = df_pos.append({'POS': pos, 'count': count, 'word': alice_doc.vocab.strings[orth_id]}, ignore_index=True)
        #print(pos, count, alice_doc.vocab.strings[orth_id])
df_pos['text_source'] = 'Carroll'

for token in persuasion_doc:
    pos_counts[token.pos][token.orth] += 1

for pos_id, counts in sorted(pos_counts.items()):
    pos = persuasion_doc.vocab.strings[pos_id]
    for orth_id, count in counts.most_common():
        df_pos = df_pos.append({'POS': pos, 
                                'count': count, 
                                'word': alice_doc.vocab.strings[orth_id], 
                                'text_source': 'Austen'}, ignore_index=True)


Y = df_pos['text_source']
X = df_pos.drop(columns=['text_source', 'word', 'POS' ])
X = pd.concat([X, pd.get_dummies(df_pos['POS'])], axis=1)
X = pd.concat([X, pd.get_dummies(df_pos['word'])], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.4,
                                                    random_state=0)

#%%
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # No need to specify l2 as it's the default. But we put it for demonstration.
train = lr.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', lr.score(X_train, y_train))
print('\nTest set score:', lr.score(X_test, y_test))
score = cross_val_score(lr, X, Y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("LR: Input X --> %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

#%% [markdown]
# #### Outcome: Bad model performance --> no further investigation for this approach
#%% [markdown]
# ### Contextual information (length of previous and next sentences, words repeated from one sentence to the next, etc),
#%%
# Load the spacy model that you have installed
nlp_core = spacy.load('en_core_web_md')

doc = nlp_core("This is some text that I am processing with Spacy")


df_vec = pd.DataFrame(columns=['sent', 'text_source'])
for sent, author in alice_sents:
    df_corpus = df_corpus.append({'sent': sent.text, 
                                'text_source': author}, ignore_index=True)
for sent, author in persuasion_sents:
    df_corpus = df_corpus.append({'sent': sent.text, 
                                'text_source': author}, ignore_index=True)
#%% [markdown]
# ### TFIDF
#%%
df_corpus = pd.DataFrame(columns=['sent', 'text_source'])
for sent, author in alice_sents:
    df_corpus = df_corpus.append({'sent': sent.text, 
                                'text_source': author}, ignore_index=True)
for sent, author in persuasion_sents:
    df_corpus = df_corpus.append({'sent': sent.text, 
                                'text_source': author}, ignore_index=True)

# Define X and y
X = TfidfVectorizer().fit_transform(df_corpus.sent)
Y = df_corpus['text_source']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y,
                                                    test_size=0.4,
                                                    random_state=0)

#%%
lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000) # No need to specify l2 as it's the default. But we put it for demonstration.
train = lr.fit(X_train, y_train)
print(X_train.shape, y_train.shape)
print('Training set score:', lr.score(X_train, y_train))
print('\nTest set score:', lr.score(X_test, y_test))
score = cross_val_score(lr, X, Y, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
print("LR: Input X --> %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
#%% [markdown]
# # Challenge 1:
# Find out whether your new model is good at identifying Alice in Wonderland vs any other work, Persuasion vs any other work, or Austen vs any other work.  This will involve pulling a new book from the Project Gutenberg corpus (print(gutenberg.fileids()) for a list) and processing it.
# 
# Record your work for each challenge in a notebook and submit it below.

#%%



