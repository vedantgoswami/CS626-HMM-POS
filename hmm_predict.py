import numpy as np
import pickle
import streamlit as st
import string


# Load the saved components
with open('pickle_files_backup/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('pickle_files_backup/tags.pkl', 'rb') as f:
    tags = pickle.load(f)

with open('pickle_files_backup/initial_probabilities.npy', 'rb') as f:
    initial_probabilities = np.load(f)

with open('pickle_files_backup/transition_matrix.npy', 'rb') as f:
    transition_matrix = np.load(f)

with open('pickle_files_backup/emission_matrix.npy', 'rb') as f:
    emission_matrix = np.load(f)


tag_colors = {
    "ADJ": "#DB7093",  # Darker Pink
    "ADP": "#4682B4",  # Darker Blue
    "CONJ": "#DAA520", # Darker Gold
    "DET": "#8FBC8F",  # Darker Green
    "NOUN": "#CD5C5C", # Darker Salmon
    "NUM": "#BA55D3",  # Darker Orchid
    "PRON": "#FF4500", # Darker Tomato
    "PRT": "#708090",  # Slate Gray
    "VERB": "#4682B4", # Darker Sky Blue
    "X": "#A9A9A9"     # Darker Gray
}

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

def predict_pos_tags(sentence, vocab, tags, initial_probabilities, transition_matrix, emission_matrix):
    sentence = [(word if word in vocab else assign_unk(word), None) for word in sentence]
    best_probs, best_paths = initialize(tag_counts=None, A=transition_matrix, B=emission_matrix, sentence=sentence, vocab=vocab)
    best_probs, best_paths = viterbi_forward(transition_matrix, emission_matrix, sentence, best_probs, best_paths, vocab, tag_counts=None)
    pred = viterbi_backward(best_probs, best_paths, tag_counts=None)
    return pred

if __name__ == "__main__":
    st.title("Assignment1: HMM-based POS Tagging")
    # user_sentence = "The quick brown fox jumps over the lazy dog"
    user_sentence = st.text_input("Enter a sentence:", "This demo showcases Assignment 1, which focuses on HMM based POS tagging for CS626 course.")
    if st.button("Predict POS Tags"):
        # Preprocess the sentence (e.g., lowercasing and tokenization)
        processed_sentence = user_sentence.lower().split()
        
        # Predict the POS tags
        predicted_tags = predict_pos_tags(processed_sentence, vocab, tags, initial_probabilities, transition_matrix, emission_matrix)
        
        # Display the results in a single line
        st.write("### Tagged Sentence:")
        html_content = "<div style='display: flex; align-items: flex-start;'>"
        for word, tag in zip(processed_sentence, predicted_tags):
            word_html = f"<div style='text-align: center; margin: 0 10px;'>"
            word_html += f"<span style='background-color: white; color: black; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{word}</span><br>"
            word_html += f"<span style='background-color: {tag_colors.get(tag, '#E0E0E0')}; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{tag}</span>"
            word_html += "</div>"
            html_content += word_html
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)

