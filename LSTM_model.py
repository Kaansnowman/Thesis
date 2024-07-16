import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import keras
import tf_keras
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.preprocessing.sequence import pad_sequences
from tf_keras.models import Sequential
from tf_keras.layers import LSTM, Dense, Embedding, Bidirectional, Conv1D, MaxPooling1D
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint

# Data Loading
reviews = pd.read_csv('sample2.csv')

# Data Preprocessing
reviews['sentiment'] = reviews['sentiment'].map({1: 0, 2: 1})
sentences = reviews['review'].to_numpy()
labels = reviews['sentiment'].to_numpy()

# Train Test Data Split
X_train, X_temp, y_train, y_temp = train_test_split(sentences, labels, test_size=0.20)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Tokenizer
vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

# Tokenize Train Data
sequence_length = 80
train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')

# Tokenizing Validation and Test Data
val_sequences = tokenizer.texts_to_sequences(X_val)
val_padded = pad_sequences(val_sequences, maxlen=sequence_length, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')

# Model Configuration
num_words = 10000
embeddings = 256
inp_length = 80

model = tf_keras.Sequential()
model.add(Embedding(num_words,embeddings,input_length=inp_length))
model.add(Conv1D(256,5,activation='relu'))
model.add(tf_keras.layers.Dropout(0.3))
model.add(MaxPooling1D(5))
model.add(tf_keras.layers.Dropout(0.3))
model.add(tf_keras.layers.Bidirectional(LSTM(128,return_sequences=True)))
model.add(LSTM(128))
model.add(tf_keras.layers.Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
optimizer = tf_keras.optimizers.Adam(learning_rate=0.0025)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

# Callbacks
checkpoint_filepath = os.getcwd()
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)

# Model Training
history = model.fit(train_padded, y_train, epochs=20, validation_data=(val_padded, y_val), callbacks=[early_stopping_callback, model_checkpoint_callback])

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Confusion Matrix
y_pred = model.predict(test_padded)
y_pred = np.round(y_pred).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# F1 Score
print("F1 Score:", f1_score(y_test, y_pred))

# Save Trained Model
model.save("LSTM_GOOD_model")
