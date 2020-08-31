import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# Function to add sentence beginner
def add_sent_beginner(tagged_sents):
	ts = []
	for s in tagged_sents:
		s.insert(0, ("^", "^"))
		ts.append(s)
	return ts

# Function to separate words and tags from tagged data
def separate_words_tags(tagged_sents):
	sent_words, sent_tags = [], []
	for ts in tagged_sents:
		words, tags = zip(*ts)
		sent_words.append(np.array(words))
		sent_tags.append(np.array(tags))
	return sent_words, sent_tags

# Function to split data
def split_data(sent_words, sent_tags, test_size=0.2):
	train_sents, test_sents, train_tags, test_tags = train_test_split(sent_words, sent_tags, test_size=test_size)
	return train_sents, test_sents, train_tags, test_tags

# Function to split data kfolds
def kfold_data(sent_words, sent_tags, k=5):
	sent_words_folds = []
	set_tags_folds = []
	split = int(len(sent_words)/k)
	for i in range(k-1):
		sent_words_folds.append(sent_words[split*i:split*(i+1)])
		sent_tags_folds.append(sent_tags[split*i:split*(i+1)])
	sent_words_folds.append(sent_words[split*(k-1):])
	sent_tags_folds.append(sent_tags[split*(k-1):])
	return sent_words_folds, sent_tags_folds

# Function to generate word labels and tag labels
def label_data(sent_words, sent_tags):
	words, tags = set([]), set([])
	for s in sent_words:
		for w in s:
			# Note that we are taking lower case while labelling
			words.add(w.lower())
	for s in sent_tags:
		for t in s:
			tags.add(t)
	word2label = {w: i+2 for i,w in enumerate(list(words))}
	word2label['PAD'] = 0
	word2label['OOV'] = 1
	tag2label = {t: i+1 for i,t in enumerate(list(tags))}
	tag2label['PAD'] = 0
	return word2label, tag2label

# Convert dataset in terms of labels
# Not specific to test or train data
def apply_labels(sent_words, sent_tags, word2label, tag2label):
	sent_words_labelled, sent_tags_labelled = [], []
	for s in sent_words:
		labelled_s = []
		for w in s:
			try:
				labelled_s.append(word2label[w.lower()])
			except KeyError:
				labelled_s.append(word2label['OOV'])
		sent_words_labelled.append(labelled_s)
	for s in sent_tags:
		labelled_s = []
		for t in s:
			try:
				labelled_s.append(tag2label[t])
			except KeyError:
				print("Tag is missing, can't be the case")
		sent_tags_labelled.append(labelled_s)
	return sent_words_labelled, sent_tags_labelled
# Pad labelled sentences
# Not specific to test or train
def pad_sentences(sent_words_labelled, sent_tags_labelled, MAX_LEN):
	sent_words_labelled = pad_sequences(sent_words_labelled, maxlen = MAX_LEN, padding='post')
	sent_tags_labelled = pad_sequences(sent_tags_labelled, maxlen = MAX_LEN, padding='post')
	return sent_words_labelled, sent_tags_labelled

# Tags to one hot encodings
def one_hot(sequences, num_categories):
	one_hot_sequences = []
	for s in sequences:
		one_hot_seq = []
		for i in s:
			one_hot_seq.append(np.zeros(num_categories))
			one_hot_seq[-1][i] = 1.0
		one_hot_sequences.append(one_hot_seq)
	return np.array(one_hot_sequences)

# Convert one hot encodings to tags
def one_hot_to_tags(sequences, label2tag):
	sent_tags = []
	for s in sequences:
		sent_tag = []
		for c in s:
			label = np.argmax(c) # softmax classifier
			sent_tag.append(label2tag[label])
		sent_tags.append(sent_tag)
	return sent_tags

# Function to convert tags from labels to tags
def label_to_tag(sequences, label2tag):
	tags = []
	for seq in sequences:
		for s in seq:
			tags.append(label2tag[s])
	return tags

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy