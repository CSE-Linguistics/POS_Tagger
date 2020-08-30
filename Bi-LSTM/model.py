from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
from utils import * 
import nltk

# Get the train and test data
tagged_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')
sent_words, sent_tags = separate_words_tags(tagged_sentences)
MAX_LEN = len(max(sent_words, key=len))

# Split data
train_sent_words, test_sent_words, train_sent_tags, test_sent_tags = split_data(sent_words, sent_tags)

# Get labels, apply them
word2label, tag2label = label_data(train_sent_words, train_sent_tags)
train_sent_words_labelled, train_sent_tags_labelled = apply_labels(train_sent_words, train_sent_tags, word2label, tag2label)
test_sent_words_labelled, test_sent_tags_labelled = apply_labels(test_sent_words, test_sent_tags, word2label, tag2label)

# Pad sentences
train_sent_words_labelled, train_sent_tags_labelled = pad_sentences(train_sent_words_labelled, train_sent_tags_labelled, MAX_LEN)
test_sent_words_labelled, test_sent_tags_labelled = pad_sentences(test_sent_words_labelled, test_sent_tags_labelled, MAX_LEN)

# Initialize neural net architecture 
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LEN, )))
model.add(Embedding(len(word2label), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2label))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy', ignore_class_accuracy(0)])

# Train the model
model.fit(train_sent_words_labelled, one_hot(train_sent_tags_labelled, len(tag2label)), batch_size=128, epochs=40, validation_split=0.2)