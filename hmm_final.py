import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import nltk
from nltk.corpus import brown
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from collections import defaultdict
import pickle
import string
import csv

nltk.download('brown')
nltk.download('universal_tagset')

# Load the corpus
sentences = brown.tagged_sents(tagset='universal')

# Preprocess the sentences (convert words to lowercase)
preprocessed_sentences = []
for sentence in sentences:
    preprocessed_sentence = [(word.lower(), tag) for word, tag in sentence]
    preprocessed_sentences.append(preprocessed_sentence)

sentences = preprocessed_sentences

words = set()
vocab = {}
tags = []

# Punctuation characters
punct = set(string.punctuation)
noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]

def assign_unk(tok):
    # Digits
    if any(char.isdigit() for char in tok):
        return "--unk_digit--"
    # Punctuation
    elif any(char in punct for char in tok):
        return "--unk_punct--"
    # Nouns
    elif any(tok.endswith(suffix) for suffix in noun_suffix):
        return "--unk_noun--"
    # Verbs
    elif any(tok.endswith(suffix) for suffix in verb_suffix):
        return "--unk_verb--"
    # Adjectives
    elif any(tok.endswith(suffix) for suffix in adj_suffix):
        return "--unk_adj--"
    # Adverbs
    elif any(tok.endswith(suffix) for suffix in adv_suffix):
        return "--unk_adv--"
    return "--unk--"

def create_dictionaries(training_corpus):
    global tags  # Ensure tags is accessible globally
    words = set()
    word_counts = defaultdict(int)  # First pass: count word frequencies
    for sentence in training_corpus:
        for word, tag in sentence:
            word_counts[word] += 1  # Count word occurrences

    # Second pass: Build vocabulary and calculate emission, transition, and tag counts
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    for sentence in training_corpus:
        prev_tag = '^'
        for word, tag in sentence:
            # If word appears less than 2 times, mark it as --unk--
            if word_counts[word] <= 1:
                word = assign_unk(word)
            words.add(word)
            emission_counts[(tag, word)] += 1
            transition_counts[(prev_tag, tag)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    words.add("--unk--")
    tags = sorted(tag_counts.keys())  # Store sorted tags globally

    # Build vocabulary based on words, marking low-frequency words as --unk--
    vocab = {}
    for word in sorted(words):
        vocab[word] = len(vocab)  # Assign a unique index to each word

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
        word_index = vocab.get(sentence[0][0], vocab[assign_unk(sentence[0][0])])
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
            word_index = vocab.get(word, vocab[assign_unk(word)])
            
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

def per_pos_metrics(all_true_tags, all_pred_tags):
    # Per POS Accuracy
    per_pos_accuracy = {tag: accuracy_score(
        [t == tag for t in all_true_tags],
        [p == tag for p in all_pred_tags]
    ) for tag in tags}

    print("\nPer POS Accuracy:")
    for tag, acc in per_pos_accuracy.items():
        print(f"{tag}: {acc:.4f}")
        
    # Per POS Precision
    per_pos_precision = {tag: precision_score(
        [t == tag for t in all_true_tags],
        [p == tag for p in all_pred_tags],
    ) for tag in tags}

    print("\nPer POS Precision:")
    for tag, acc in per_pos_precision.items():
        print(f"{tag}: {acc:.4f}")
        
    # Per POS Recall
    per_pos_recall = {tag: recall_score(
        [t == tag for t in all_true_tags],
        [p == tag for p in all_pred_tags], average='weighted', zero_division=0
    ) for tag in tags}

    print("\nPer POS Recall:")
    for tag, acc in per_pos_recall.items():
        print(f"{tag}: {acc:.4f}")
        
    # Per POS F1 score
    per_pos_f1 = {tag: f1_score(
        [t == tag for t in all_true_tags],
        [p == tag for p in all_pred_tags], average='weighted', zero_division=0
    ) for tag in tags}

    print("\nPer POS F1 score:")
    for tag, acc in per_pos_f1.items():
        print(f"{tag}: {acc:.4f}")

def save_pickles(fold):
    # Save the components to files
    with open(f'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    with open(f'tags.pkl', 'wb') as f:
        pickle.dump(tags, f)

    with open(f'initial_probabilities.npy', 'wb') as f:
        np.save(f, initial_probabilities)

    with open(f'transition_matrix.npy', 'wb') as f:
        np.save(f, transition_matrix)

    with open(f'emission_matrix.npy', 'wb') as f:
        np.save(f, emission_matrix)

def overall_metric(all_true_tags, all_pred_tags):
    # Overall Accuracy
    overall_accuracy = accuracy_score(all_true_tags, all_pred_tags)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    # Overall Recall
    overall_recall = recall_score(all_true_tags, all_pred_tags, average='weighted', zero_division=0)
    print(f"\nOverall Recall: {overall_recall:.4f}")

    # Overall Precision
    overall_precision = precision_score(all_true_tags, all_pred_tags, average='weighted', zero_division=0)
    print(f"\nOverall Precision: {overall_precision:.4f}")

    # Overall F_1 score
    overall_f_one  = f1_score(all_true_tags, all_pred_tags, average='weighted', zero_division=0)
    print(f"\nOverall F_1: {overall_f_one:.4f}")

    # Overall F_half score
    overall_f_half  = fbeta_score(all_true_tags, all_pred_tags, beta=0.5, average='weighted', zero_division=0)
    print(f"\nOverall F_0.5: {overall_f_half:.4f}")

    # Overall F_Two score
    overall_f_two  = fbeta_score(all_true_tags, all_pred_tags, beta=2, average='weighted', zero_division=0)
    print(f"\nOverall F_2: {overall_f_two:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_true_tags, all_pred_tags, labels=tags)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Normalize the confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(conf_matrix_normalized, index=tags, columns=tags)

    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='.2f', cmap="YlGnBu")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.savefig('./results/output_normalized.png')
    plt.show()

def save_wrong_prediction(wrong_predictions, fold):
    # Save wrong predictions to CSV
    csv_filename = f'./error_analysis/wrong_predictions_fold_{fold + 1}.csv'
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Sentence', 
            'Actual_Sentence_Tag', 
            'Predicted_Sentence_Tag', 
            'Actual_Tags', 
            'Predicted_Tags', 
            'Actual_Pred_Wrong_Tag', 
            'Num_UNK_Noun', 
            'Num_Wrong_UNK_Noun', 
            'Num_UNK_Verb', 
            'Num_Wrong_UNK_Verb', 
            'Num_UNK_Adj', 
            'Num_Wrong_UNK_Adj', 
            'Num_UNK_Adv', 
            'Num_Wrong_UNK_Adv', 
            'Num_UNK_Punct', 
            'Num_Wrong_UNK_Punct', 
            'Num_UNK_Digit', 
            'Num_Wrong_UNK_Digit', 
            'Num_UNK_General', 
            'Num_Wrong_UNK_General'
        ])
        writer.writerows(wrong_predictions)
        print(f"\nWrong predictions saved to {csv_filename}")

def count_unk_categories(sentence, vocab):
    counts = {
        "--unk_punct--":0,
        "--unk_digit--":0,
        "--unk_noun--": 0,
        "--unk_verb--": 0,
        "--unk_adj--": 0,
        "--unk_adv--": 0,
        "--unk--": 0
    }
    
    for word, tag in sentence:
        # Check if the word is in the vocabulary
        if word in counts.keys():
            unk_category = word
            if unk_category in counts:
                counts[unk_category] += 1
        # If it's in vocab, we do nothing (as it's not considered an unknown)
    return counts

def count_wrong_unk_categories(sentence, true_tags, pred_tags, vocab):
    wrong_counts = {
        "--unk_punct--": 0,
        "--unk_digit--": 0,
        "--unk_noun--": 0,
        "--unk_verb--": 0,
        "--unk_adj--": 0,
        "--unk_adv--": 0,
        "--unk--": 0
    }
    
    for i, (word, tag) in enumerate(sentence):
        # Check if the word is in the vocabulary
        if word  in wrong_counts.keys():
            unk_category = word
            # Only count as wrong if the true tag doesn't match the predicted tag
            if unk_category in wrong_counts and true_tags[i] != pred_tags[i]:
                wrong_counts[unk_category] += 1
                
    return wrong_counts


# 5-fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alpha = 0.001

all_true_tags = []
all_pred_tags = []
fold_accuracies = []
wrong_predictions = []

best_accuracy = 0
best_fold = -1

for fold, (train_index, test_index) in enumerate(kf.split(sentences)):
    wrong_predictions = []
    print(f"\nProcessing Fold {fold + 1}...")
    
    train_sentences = [sentences[i] for i in train_index]
    test_sentences = [sentences[i] for i in test_index]
    
    emission_counts, transition_counts, tag_counts, vocab = create_dictionaries(train_sentences)
    transition_matrix = create_transition_matrix(alpha, tag_counts, transition_counts)
    emission_matrix = create_emission_matrix(alpha, tag_counts, emission_counts, vocab)
    initial_probabilities = create_initial_matrix(tag_counts, transition_counts)
    
    print(f"\nRequired matrices created...")
    
    fold_true_tags = []
    fold_pred_tags = []
    
    for sentence in test_sentences:
        true_tags = [tag for word, tag in sentence]
        processed_sentence = [(word if word in vocab else assign_unk(word), tag) for word, tag in sentence]
        best_probs, best_paths = initialize(tag_counts, transition_matrix, emission_matrix, processed_sentence, vocab)
        best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, processed_sentence, best_probs, best_paths, vocab, tag_counts)
        pred = viterbi_backward(best_probs, best_paths, tag_counts)
        
        fold_true_tags.extend(true_tags)
        fold_pred_tags.extend(pred)
        
        # Check if there are any incorrect predictions in the sentence
        # Check if there are any incorrect predictions in the sentence
        if true_tags != pred:
            # Format sentence with tags for actual and predicted
            actual_sentence = " ".join([f"{word}" for (word, tag) in sentence])
            actual_sentence_tag = ", ".join([f"{word}_{tag}" for (word, tag) in sentence])
            
            # Maintain unk words in predicted_sentence_tag
            predicted_sentence_tag = ", ".join(
                [f"{word if word in vocab else assign_unk(word)}_{pred[i]}" for i, (word, tag) in enumerate(sentence)]
            )
            
            actual_tag = ", ".join([f"{tag}" for (word, tag) in sentence])
            predicted_tag = ", ".join([f"{pred[i]}" for i, (word, tag) in enumerate(sentence)])
            actual_pred_wrong_tag = ", ".join(
                [f"{true_tags[i]}_{pred[i]}" for i in range(len(true_tags)) if true_tags[i] != pred[i]]
            )
            # Count the number of each specific type of --unk-- words
            unk_counts = count_unk_categories(processed_sentence,vocab)
            # Count the number of wrong predictions for each specific type of --unk-- words
            wrong_unk_counts = count_wrong_unk_categories(processed_sentence, true_tags, pred,vocab)
            
            
            # Append the wrong prediction details
            # Append the wrong prediction details
            wrong_predictions.append([
                        actual_sentence, 
                        actual_sentence_tag, 
                        predicted_sentence_tag, 
                        actual_tag, 
                        predicted_tag, 
                        actual_pred_wrong_tag,
                        unk_counts['--unk_noun--'],  # UNK Noun Count
                        wrong_unk_counts['--unk_noun--'],  # Wrong UNK Noun Count
                        unk_counts['--unk_verb--'],  # UNK Verb Count
                        wrong_unk_counts['--unk_verb--'],  # Wrong UNK Verb Count
                        unk_counts['--unk_adj--'],  # UNK Adjective Count
                        wrong_unk_counts['--unk_adj--'],  # Wrong UNK Adjective Count
                        unk_counts['--unk_adv--'],  # UNK Adverb Count
                        wrong_unk_counts['--unk_adv--'],  # Wrong UNK Adverb Count
                        unk_counts['--unk_punct--'],  # UNK Punctuation Count
                        wrong_unk_counts['--unk_punct--'],  # Wrong UNK Punctuation Count
                        unk_counts['--unk_digit--'],  # UNK Digit Count
                        wrong_unk_counts['--unk_digit--'],  # Wrong UNK Digit Count
                        unk_counts['--unk--'],  # General UNK Count
                        wrong_unk_counts['--unk--']  # Wrong General UNK Count
                    ])
    
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
    
    save_wrong_prediction(wrong_predictions, fold=fold)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_fold = fold
        # Save pickle files for the best performing fold
        save_pickles(fold=best_fold)

per_pos_metrics(all_true_tags, all_pred_tags)
overall_metric(all_true_tags, all_pred_tags)
