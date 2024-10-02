import numpy as np
import pickle
import streamlit as st
import os
import string
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')  # Only needs to be run once
def tokenize_sentence(sentence):
    return word_tokenize(sentence)

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


# Get the current working directory
current_dir = os.getcwd()

# Define the relative path to the demo_files directory
demo_files_dir = os.path.join(current_dir, 'demo_files')

# Check if we're already in the demo_files directory
if os.path.basename(current_dir) == 'demo_files':
    demo_files_dir = current_dir  # We're already inside demo_files


noun_suffix = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_suffix = ["ate", "ify", "ise", "ize", "ed", "ing"]
adj_suffix = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ous"]
adv_suffix = ["ward", "wards", "wise", "ly"]
punct = set(string.punctuation)

def word_features(sentence, i):
    word = sentence[i]
    prevword = sentence[i-1] if i > 0 else '<START>'
    nextword = sentence[i+1] if i < len(sentence)-1 else '<END>'

    features = {
        'word': word,
        
        'is_numeric': word.isdigit(),
        'contains_number': any(char.isdigit() for char in word),
        
        'is_punctuation': any(char in punct for char in word),
        
        # prefix suffix features
        'has_noun_suffix': any(word.endswith(suffix) for suffix in noun_suffix),
        'has_verb_suffix': any(word.endswith(suffix) for suffix in verb_suffix),
        'has_adj_suffix': any(word.endswith(suffix) for suffix in adj_suffix),
        'has_adv_suffix': any(word.endswith(suffix) for suffix in adv_suffix),
        
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:],
        
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
 

def predict_pos_tags(sentence):
    # Load the saved components
    with open(os.path.join(demo_files_dir, 'best_crf_model.pkl'), 'rb') as f:
        loaded_crf = pickle.load(f)
    pred = loaded_crf.predict([sentence])
    pred =[label for sublist in pred for label in sublist]
    return pred

if __name__ == "__main__":
    st.title("Assignment1: CRF-based POS Tagging")
    # user_sentence = "The quick brown fox jumps over the lazy dog"
    sentence = st.text_input("Enter a sentence:", "They refuse to go")
    if st.button("Predict POS Tags"):
        # Preprocess the sentence (e.g., lowercasing and tokenization)
        X_sentence = []
        tokens = tokenize_sentence(sentence)

        # Create feature set for each word
        for i in range(len(tokens)):
            X_sentence.append(word_features(tokens, i))
            
        # Predict the POS tags
        predicted_tags = predict_pos_tags(X_sentence)
        
        # Display the results in a single line
        st.write("### Tagged Sentence:")
        html_content = "<div style='display: flex; align-items: flex-start;'>"
        for i in range(len(tokens)):
            word = tokens[i]
            tag = predicted_tags[i]
            print(word,tag)
            word_html = f"<div style='text-align: center; margin: 0 10px;'>"
            word_html += f"<span style='background-color: white; color: black; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{word}</span><br>"
            word_html += f"<span style='background-color: {tag_colors.get(tag, '#E0E0E0')}; color: white; font-weight: bold; padding: 5px 10px; border-radius: 5px;'>{tag}</span>"
            word_html += "</div>"
            html_content += word_html
        html_content += "</div>"
        st.markdown(html_content, unsafe_allow_html=True)

