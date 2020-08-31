from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
from utils import * 
import nltk
from sklearn.metrics import confusion_matrix

# Get the train and test data
tagged_sentences = nltk.corpus.brown.tagged_sents(tagset='universal')
tagged_sentences = add_sent_beginner(tagged_sentences)
sent_words, sent_tags = separate_words_tags(tagged_sentences)
MAX_LEN = len(max(sent_words, key=len))

# 5-fold cross validation
k = 5
acc_per_fold = []
loss_per_fold = []
sent_words_folds, sent_tags_folds = kfold_data(sent_words, sent_tags)
for i in range(k):
	print("Fold number {}".format(i+1))
	test_sent_words = sent_words_folds[i]
	test_sent_tags = sent_tags_folds[i]
	train_sent_words = [] 
	train_sent_tags = []
	for j in range(k):
		if j!=i:
			train_sent_words += sent_words_folds[j] 
			train_sent_tags += 	sent_tags_folds[j]


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
	model.fit(train_sent_words_labelled, one_hot(train_sent_tags_labelled, len(tag2label)), batch_size=128, epochs=2)
	scores = model.evaluate(test_sent_words_labelled, one_hot(test_sent_tags_labelled, len(tag2label)), verbose=0)
	acc_per_fold.append(scores[1] * 100)
	loss_per_fold.append(scores[0])

	# Print confusion matrix
	predictions = model.predict(test_sent_words_labelled)
	predicted_tags = one_hot_to_tags(predictions, {i: t for t, i in tag2label.items()})
	concatenated_predictions = []
	concatenated_tests = label_to_tag(test_sent_tags_labelled, {i: t for t, i in tag2label.items()})
	for pt in predicted_tags:
		for t in pt:
			concatenated_predictions.append(t) 
	print(confusion_matrix(concatenated_tests, concatenated_predictions))