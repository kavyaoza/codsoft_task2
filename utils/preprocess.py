# utils/preprocess.py

import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Clean caption text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.strip()
    text = ' '.join(text.split())  # remove multiple spaces
    return text

# Load and clean captions from dictionary
def preprocess_captions(captions_dict):
    cleaned_captions = {}
    for img_id, captions in captions_dict.items():
        cleaned = [f"<start> {clean_text(c)} <end>" for c in captions]
        cleaned_captions[img_id] = cleaned
    return cleaned_captions

# Fit tokenizer on all captions
def create_tokenizer(cleaned_captions):
    all_captions = []
    for captions in cleaned_captions.values():
        all_captions.extend(captions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# Convert captions to sequences
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = np.zeros(vocab_size)
            out_seq[out_seq] = 1.0
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)
