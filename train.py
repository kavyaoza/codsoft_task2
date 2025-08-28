# train.py

import os
import numpy as np
import string
import pickle
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.callbacks import ModelCheckpoint

# ----- Load captions -----
def load_captions(filename):
    captions = {}
    with open(filename, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) < 2:
                continue
            img_id, caption = tokens[0].split('#')[0], tokens[1]
            if img_id not in captions:
                captions[img_id] = []
            # ✅ USE startseq and endseq instead of <start> and <end>
            captions[img_id].append('startseq ' + clean_text(caption) + ' endseq')
    return captions

# ----- Clean text -----
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    text = ' '.join(text.split())
    return text

# ----- Load images and extract features -----
def extract_features(directory):
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features = {}
    for img_name in tqdm(os.listdir(directory)):
        filename = os.path.join(directory, img_name)
        img = image.load_img(filename, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature
    return features

# ----- Prepare tokenizer -----
def create_tokenizer(captions_dict):
    all_captions = []
    for cap_list in captions_dict.values():
        all_captions.extend(cap_list)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

# ----- Create sequences -----
def create_sequences(tokenizer, max_length, captions_dict, features, vocab_size):
    X1, X2, y = [], [], []
    for key, desc_list in captions_dict.items():
        if key not in features:
            continue  # Skip missing images
        photo = features[key][0]
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)


# ----- Define model -----
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# ===== Main Training =====
if __name__ == "__main__":
    dataset_path = "Flickr8k_Dataset"
    captions_path = "Flickr8k_text/Flickr8k.token.txt"

    print("Loading and cleaning captions...")
    captions = load_captions(captions_path)

    print("Extracting image features...")
    features = extract_features(dataset_path)

    print("Creating tokenizer...")
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 35

    print("Creating sequences...")
    X1, X2, y = create_sequences(tokenizer, max_length, captions, features, vocab_size)

    print("Defining model...")
    model = define_model(vocab_size, max_length)

    print("Training model...")
    filepath = 'caption_model_weights.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit([X1, X2], y, epochs=10, verbose=1, callbacks=[checkpoint])

    print("Saving tokenizer...")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("Training complete! ✔️")
import pickle

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("✅ Saved tokenizer to tokenizer.pkl")

# Save model weights
model.save_weights('caption_model_weights.h5')
print("✅ Saved model weights to caption_model_weights.h5")
