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

	# Print confusion matrix
	predictions = model.predict(test_sent_words_labelled)
	predicted_tags = one_hot_to_tags(predictions, {i: t for t, i in tag2label.items()})
	concatenated_predictions = []
	concatenated_tests = label_to_tag(test_sent_tags_labelled, {i: t for t, i in tag2label.items()})
	for pt in predicted_tags:
		for t in pt:
			concatenated_predictions.append(t) 
	conf_mat = confusion_matrix(concatenated_tests, concatenated_predictions)
	acc = sum(np.diag(conf_mat))/sum(sum(conf_mat[:,:]))
	acc_per_fold.append(acc)

	## Code to analyse errors in detail:
	# concatenated_words = label_to_tag(test_sent_words_labelled, {i: w for w, i in word2label.items()})
	# word_original_tag = defaultdict(int)
	# word_predicted_tag = defaultdict(int)
	# word_otag_ptag = defaultdict(int)
	# for iter in range(len(concatenated_predictions)):
	# 	if concatenated_predictions[iter] != concatenated_tests[iter]:
	# 		word_original_tag[(concatenated_words[iter], concatenated_tests[iter])] += 1
	# 		word_predicted_tag[(concatenated_words[iter], concatenated_predictions[iter])] += 1
	# 		word_otag_ptag[(concatenated_words[iter], concatenated_tests[iter], concatenated_predictions[iter])] += 1
	# sort_word_otag = sorted(word_original_tag.items(), key=lambda x: x[1], reverse=True)
	# sort_word_ptag = sorted(word_predicted_tag.items(), key=lambda x: x[1], reverse=True)
	# sort_word_otag_ptag = sorted(word_otag_ptag.items(), key=lambda x: x[1], reverse=True)
	# print(sort_word_otag)
	# print(sort_word_ptag)
	# print(sort_word_otag_ptag)