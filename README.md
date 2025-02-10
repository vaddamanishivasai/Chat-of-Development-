# Chat-of-Development-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sample conversational data
conversations = [
    "Hi, how are you?",
    "I'm good, thank you. How about you?",
    "I'm great, thanks for asking!",
    "What can I help you with today?",
    "I have a question about my order.",
    "Sure, what's your order number?"
]
# Tokenize the conversations
tokenizer = Tokenizer()
tokenizer.fit_on_texts(conversations)
sequences = tokenizer.texts_to_sequences(conversations)
vocab_size = len(tokenizer.word_index) + 1
# Pad the sequences
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
# Create the LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64)),
    Dense(vocab_size, activation='softmax')
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Prepare the input and output sequences
X = padded_sequences[:-1]
y = padded_sequences[1:]
y = np.expand_dims(y, axis=-1)
# Train the model
model.fit(X, y, epochs=100, verbose=1)
# Generate responses
def generate_response(model, tokenizer, input_text, max_length):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')
    prediction = model.predict(input_sequence)
    predicted_word_index = np.argmax(prediction)
    predicted_word = tokenizer.index_word[predicted_word_index]
    return predicted_word
# Example usage
input_text = "Hi, how are you?"
response = generate_response(model, tokenizer, input_text, max_length)
print("Response:", response)
