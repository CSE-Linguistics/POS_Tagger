import nltk
import numpy as np

# Load tagged corpora
tagged_data = nltk.corpus.brown.tagged_words(tagset='universal')

# Find all unique tags and count for each tag
tags_all = []
for (word,tag) in tagged_data:
	tags_all.append(tag)

(tagset, count) = np.unique(tags_all, return_counts=True)

tags_dict = {}
for i in range(len(tagset)):
	tags_dict[tagset[i]] = count[i]

# Find all unique words
words_all = nltk.corpus.brown.words()
words = np.unique(words_all)

# Find count of all (word, tag pairs)
word_tag_freq = {}
for w in words:
	for t in tagset:
		word_tag_freq[(w,t)] = 0
for (w,t) in tagged_data:
	word_tag_freq[(w,t)] += 1

# Find count of one tag following another
# t1 is the first tag, t2 is the following tag
transitions = {}
for t1 in tagset:
	for t2 in tagset:
		transitions[(t1,t2)] = 0
for i in range(len(tagged_data)-1):
	transitions[(tagged_data[i][1], tagged_data[i+1][1])] += 1

# Obtain the emission and transition probabilities
emission = word_tag_freq
for (w,t) in emission.keys():
	emission[(w,t)] = emission[(w,t)]/tags_dict[t]
for (t1,t2) in transitions.keys():
	transitions[(t1,t2)] = transitions[(t1,t2)]/tags_dict[t1]