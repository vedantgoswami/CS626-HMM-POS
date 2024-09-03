import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import pickle

nltk.download('brown')
nltk.download('universal_tagset')

# Load the corpus
sentences = brown.tagged_sents(tagset='universal')

words = set()
vocab = {}
tags = []

def create_dictionaries(training_corpus):
    global tags  # Ensure tags is accessible globally
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for sentence in training_corpus:
        prev_tag = '^'
        for word, tag in sentence:
            words.add(word)
            emission_counts[(tag, word)] += 1
            transition_counts[(prev_tag, tag)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    words.add("--unk--")
    tags = sorted(tag_counts.keys())  # Store sorted tags globally

    for i, word in enumerate(sorted(words)):
        vocab[word] = i
    return emission_counts, transition_counts, tag_counts, vocab

def create_transition_matrix(alpha, tag_counts, transition_counts):
    transition_matrix = np.zeros((len(tags), len(tags)))
    for i in range(len(tags)):
        for j in range(len(tags)):
            transition_matrix[i][j] = (transition_counts[(tags[i], tags[j])] + alpha) / (tag_counts[tags[i]] + len(tags) * alpha)
    return transition_matrix

def create_emission_matrix(alpha, tag_counts, emission_counts, vocab):
    emission_matrix = np.zeros((len(tags), len(vocab)))
    for i in range(len(tags)):
        for word in vocab:
            emission_matrix[i][vocab[word]] = (emission_counts[(tags[i], word)] + alpha) / (tag_counts[tags[i]] + len(vocab) * alpha)
    return emission_matrix

def create_initial_matrix(tag_counts, transition_counts):
    initial_probabilities = np.zeros((len(tags), 1))
    total = 0
    for i, tag in enumerate(tags):
        total += transition_counts[('^', tag)]
        initial_probabilities[i] = transition_counts[('^', tag)]
    return initial_probabilities / total

def initialize(tag_counts, A, B, sentence, vocab):
    best_probs = np.zeros((len(tags), len(sentence)))
    best_paths = np.zeros((len(tags), len(sentence)), dtype=int)
    
    for i, tag in enumerate(tags):
        word_index = vocab.get(sentence[0][0], vocab["--unk--"])
        
        if A[0, i] == 0:
            best_probs[i, 0] = float('-inf')
        else:
            best_probs[i, 0] = np.log(initial_probabilities[i][0]) + np.log(B[i][word_index])
    
    return best_probs, best_paths

def viterbi_forward(A, B, sentence, best_probs, best_paths, vocab, tag_counts):
    for i in range(1, len(sentence)): 
        for j, tag_j in enumerate(tags):
            best_prob_i = float("-inf")
            best_path_i = None
            
            word = sentence[i][0]
            word_index = vocab.get(word, vocab["--unk--"])
            
            for k, tag_k in enumerate(tags):
                prob = best_probs[k, i-1] + np.log(A[k, j]) + np.log(B[j][word_index])
                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
                    
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i
                    
    return best_probs, best_paths

def viterbi_backward(best_probs, best_paths, tag_counts):
    m = best_paths.shape[1]
    z = [None] * m
    pred = [None] * m
    
    best_prob_for_last_word = float('-inf')
    
    for k in range(len(tags)):
        if best_probs[k, -1] > best_prob_for_last_word:
            best_prob_for_last_word = best_probs[k, -1]
            z[m - 1] = k
            
    pred[m - 1] = tags[z[m - 1]]
    
    for i in range(m-1, 0, -1):
        z[i - 1] = best_paths[z[i], i]
        pred[i - 1] = tags[z[i - 1]]
    
    return pred

# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alpha = 0.002

all_true_tags = []
all_pred_tags = []
fold_accuracies = []

for fold, (train_index, test_index) in enumerate(kf.split(sentences)):
    print(f"\nProcessing Fold {fold + 1}...")
    
    train_sentences = [sentences[i] for i in train_index]
    test_sentences = [sentences[i] for i in test_index]
    
    emission_counts, transition_counts, tag_counts, vocab = create_dictionaries(train_sentences)
    transition_matrix = create_transition_matrix(alpha, tag_counts, transition_counts)
    emission_matrix = create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
    initial_probabilities = create_initial_matrix(tag_counts, transition_counts)
    
    print(f"\nrequired matrices created...")
    
    fold_true_tags = []
    fold_pred_tags = []
    
    for sentence in test_sentences:
        true_tags = [tag for word, tag in sentence]
        processed_sentence = [(word if word in vocab else "--unk--", tag) for word, tag in sentence]
        best_probs, best_paths = initialize(tag_counts, transition_matrix, emission_matrix, processed_sentence, vocab)
        best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, processed_sentence, best_probs, best_paths, vocab, tag_counts)
        pred = viterbi_backward(best_probs, best_paths, tag_counts)
        
        fold_true_tags.extend(true_tags)
        fold_pred_tags.extend(pred)
    
    accuracy = accuracy_score(fold_true_tags, fold_pred_tags)
    fold_accuracies.append(accuracy)
    all_true_tags.extend(fold_true_tags)
    all_pred_tags.extend(fold_pred_tags)
    
    print(f"Accuracy for Fold {fold + 1}: {accuracy:.4f}")
    
    # Calculate and print accuracy per POS tag for this fold
    fold_per_pos_accuracy = {tag: accuracy_score(
        [t == tag for t in fold_true_tags],
        [p == tag for p in fold_pred_tags]
    ) for tag in tags}
    
    print(f"\nPer POS Accuracy for Fold {fold + 1}:")
    for tag, acc in fold_per_pos_accuracy.items():
        print(f"{tag}: {acc:.4f}")

# Overall Accuracy
overall_accuracy = accuracy_score(all_true_tags, all_pred_tags)
print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(all_true_tags, all_pred_tags, labels=tags)
print("\nConfusion Matrix:")
print(conf_matrix)

# Per POS Accuracy
per_pos_accuracy = {tag: accuracy_score(
    [t == tag for t in all_true_tags],
    [p == tag for p in all_pred_tags]
) for tag in tags}

print("\nPer POS Accuracy:")
for tag, acc in per_pos_accuracy.items():
    print(f"{tag}: {acc:.4f}")

# Confusion Matrix
df_cm = pd.DataFrame(conf_matrix, index=tags, columns=tags)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig('output.png')
plt.show()

# saving the components 
import pickle

# Save the components to files
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('tags.pkl', 'wb') as f:
    pickle.dump(tags, f)

with open('initial_probabilities.npy', 'wb') as f:
    np.save(f, initial_probabilities)

with open('transition_matrix.npy', 'wb') as f:
    np.save(f, transition_matrix)

with open('emission_matrix.npy', 'wb') as f:
    np.save(f, emission_matrix)


