import sklearn_crfsuite
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, classification_report, make_scorer
from sklearn_crfsuite import metrics
import scipy.stats
import string
import pickle

nltk.download('brown')
nltk.download('universal_tagset')

# Load the corpus
sentences = brown.tagged_sents(tagset='universal')

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)

def word_features(sentence, i):
    word = sentence[i][0]
    prevword = sentence[i-1][0] if i > 0 else '<START>'
    nextword = sentence[i+1][0] if i < len(sentence)-1 else '<END>'

    features = {
        'word': word,
        'is_numeric': word.isdigit(),
        'contains_number': any(char.isdigit() for char in word),
        'is_punctuation': any(char in punct for char in word),
        
        # Prefix-suffix features
        'has_noun_suffix': any(word.endswith(suffix) for suffix in noun_suffix),
        'has_verb_suffix': any(word.endswith(suffix) for suffix in verb_suffix),
        'has_adj_suffix': any(word.endswith(suffix) for suffix in adj_suffix),
        'has_adv_suffix': any(word.endswith(suffix) for suffix in adv_suffix),
        
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        'prefix-2': word[:3],
        'suffix-2': word[-3:],
        
        
        'prevword': prevword,
        'nextword': nextword,
        
        # Case features
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'is_all_lower': word.islower(),
        'prevword_is_capitalized': prevword[0].isupper() if prevword != '<START>' else False,
        'nextword_is_capitalized': nextword[0].isupper() if nextword != '<END>' else False,
        
        # Length features
        'word_length': len(word),
        'prevword_length': len(prevword) if prevword != '<START>' else 0,
        'nextword_length': len(nextword) if nextword != '<END>' else 0,
        
        # Position features
        'is_first': i == 0,
    }

    return features

# Preparing dataset
X = []
y = []
for sentence in sentences:
    X_sentence = []
    y_sentence = []
    for i in range(len(sentence)):
        X_sentence.append(word_features(sentence, i))
        y_sentence.append(sentence[i][1])
    X.append(X_sentence)
    y.append(y_sentence)

# Define CRF model with fixed parameters
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

# Define the parameter space for c1 and c2
params_space = {
    'c1': scipy.stats.expon(scale=0.5),
    'c2': scipy.stats.expon(scale=0.05),
}

# Define F1 score as the evaluation metric using the weighted average
labels = [".","ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRON","PRT","VERB","X"] # Replace with known label set if necessary
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

# Perform RandomizedSearchCV with 5-fold cross-validation
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        n_iter=30,  # You can reduce this number to speed up
                        scoring=f1_scorer)

rs.fit(X,y)

print('best params:', rs.best_params_)
print('best CV score:', rs.best_score_)
print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))

# Get the best estimator
best_crf = rs.best_estimator_

# Save the model to a file using pickle
with open('best_crf_model.pkl', 'wb') as f:
    pickle.dump(best_crf, f)

print("Best CRF model saved to 'best_crf_model.pkl'")

from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(20))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-20:])

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])