import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess dataset (example: Shakespeare Sonnets)
text = "Shall I compare thee to a summer's day? Thou art more lovely and more temperate."

# Tokenize characters
chars = sorted(set(text))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for c, i in char_to_index.items()}

# Create sequences for training
seq_length = 10
sequences = []
nex_chars = []
for i in range(len(text) - seq_length):
    sequences.append([char_to_index[c] for c in text[i:i+seq_length]])
    nex_chars.append(char_to_index[text[i+seq_length]])

X = np.array(sequences)
y = tf.keras.utils.to_categorical(nex_chars, num_classes=len(chars))

# Define LSTM model
input_layer = Input(shape=(seq_length,))
embedding = Embedding(input_dim=len(chars), output_dim=8, input_length=seq_length)(input_layer)
lstm_layer = LSTM(128, return_sequences=False)(embedding)
dense_layer = Dense(len(chars), activation='softmax')(lstm_layer)

model = Model(input_layer, dense_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Generate text
def generate_text(seed_text, length=100, temperature=1.0):
    result = seed_text
    for _ in range(length):
        input_seq = np.array([[char_to_index[c] for c in result[-seq_length:]]])
        preds = model.predict(input_seq)[0]
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(len(chars), p=preds)
        next_char = index_to_char[next_index]
        result += next_char
    return result

# Example: Generate text with different temperatures
print(generate_text("Shall I com", length=200, temperature=0.5))
