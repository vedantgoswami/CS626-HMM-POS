import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
from collections import defaultdict
import string
import pickle
import csv

nltk.download('brown')
nltk.download('universal_tagset')

# Load the corpus
sentences = brown.tagged_sents(tagset='universal')

noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)

# Feature
def word_features(sentence, i):
    word = sentence[i][0]
    prevword = sentence[i-1][0] if i > 0 else '<START>'
    nextword = sentence[i+1][0] if i < len(sentence)-1 else '<END>'

    features = {
        'word': word,
        
        # numeric features
        'is_numeric': word.isdigit(),
        'contains_number': any(char.isdigit() for char in word),
        
        'is_punctuation': any(char in punct for char in word),
        
        # pos suffix features
        'has_noun_suffix': any(word.endswith(suffix) for suffix in noun_suffix),
        'has_verb_suffix': any(word.endswith(suffix) for suffix in verb_suffix),
        'has_adj_suffix': any(word.endswith(suffix) for suffix in adj_suffix),
        'has_adv_suffix': any(word.endswith(suffix) for suffix in adv_suffix),
        
        # prefix suffix features
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        
        # context features
        'prevword': prevword,
        'nextword': nextword,
        
        # case features
        'is_capitalized': word[0].isupper(),
        'is_all_caps': word.isupper(),
        'is_all_lower': word.islower(),
        
        # len features
        'word_length': len(word),
        
        # position features
        'is_first': i == 0,
    }

    return features

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

# Define number of folds
n_splits = 5
kf = KFold(n_splits=n_splits)

# Initialize variables for storing metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []
fbeta_half_scores = []
fbeta_two_scores = []
all_y_true = []
all_y_pred = []

fold = 1
for train_index, test_index in kf.split(X):
    # Split data
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    # Train CRF
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.117,
        c2=0.0440,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    # Save the model to a file using pickle
    with open(f'./demo/best_crf_model.pkl', 'wb') as f:
        pickle.dump(crf, f)
    
    # Predict
    y_pred = crf.predict(X_test)

    # Flatten the predictions and true labels for metric calculation
    y_true_flat = [label for sublist in y_test for label in sublist]
    y_pred_flat = [label for sublist in y_pred for label in sublist]
    
    # Append true and predicted labels for overall confusion matrix later
    all_y_true.extend(y_true_flat)
    all_y_pred.extend(y_pred_flat)

    # Calculate and store metrics for this fold using sklearn_crfsuite metrics
    accuracy = metrics.flat_accuracy_score(y_test, y_pred)
    precision = metrics.flat_precision_score(y_test, y_pred, average='weighted')
    recall = metrics.flat_recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted')
    fbeta_half = metrics.flat_fbeta_score(y_test, y_pred, beta=0.5, average='weighted')
    fbeta_two = metrics.flat_fbeta_score(y_test, y_pred, beta=0.5, average='weighted')
    
    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    fbeta_half_scores.append(fbeta_half)
    fbeta_two_scores.append(fbeta_two)
    
    # Print metrics for the current fold
    print(f"Fold {fold} Results:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"F-beta Score (beta=0.5): {fbeta_half}")
    print(f"F-beta Score (beta=2): {fbeta_two}")
    print("\n")
    
    fold += 1

# Calculate average metrics over all folds
print("Average Accuracy: ", np.mean(accuracies))
print("Average Precision: ", np.mean(precisions))
print("Average Recall: ", np.mean(recalls))
print("Average F1 Score: ", np.mean(f1_scores))
print("Average F-beta Score (beta=0.5): ", np.mean(fbeta_half_scores))
print("Average F-beta Score (beta=2): ", np.mean(fbeta_two_scores))

# Generate and display confusion matrix and classification report for all folds
conf_matrix = confusion_matrix(all_y_true, all_y_pred)
print("Confusion Matrix:\n", conf_matrix)

print("Classification Report:\n", classification_report(all_y_true, all_y_pred))

labels = [".","ADJ","ADP","ADV","CONJ","DET","NOUN","NUM","PRON","PRT","VERB","X",]
# Optionally, plot the confusion matrix
plt.figure(figsize=(10,7))
sn.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('./results/confusion_matrix.png')
