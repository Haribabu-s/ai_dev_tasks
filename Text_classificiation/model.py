import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Dummy medical dataset
texts = [
    "Patient reports severe headache and nausea",
    "Patient has mild cough and fever",
    "Blood pressure is stable, no major issues",
    "Patient experiencing shortness of breath",
    "Symptoms include fatigue and weight loss",
    "Patient feels energetic and healthy",
]
labels = [1, 1, 0, 1, 1, 0]  # 1 = sick, 0 = healthy

# Tokenization
vocab_size = 1000
max_length = 20
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Save tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Convert text to padded sequences
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_length, padding="post")

# Model
model = Sequential([
    Embedding(vocab_size, 16, input_length=max_length),
    LSTM(32),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(padded, np.array(labels), epochs=10, verbose=1)

# Save model
model.save("model.h5")
print("Model and tokenizer saved successfully.")
