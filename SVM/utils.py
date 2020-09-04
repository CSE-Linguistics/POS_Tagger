import numpy as np
from collections import defaultdict
import re, nltk

def k_fold_data(k : int, X: np.array):
    start_index = 0
    end_index = k
    k_folds = []

    for i in range(k):
        k_folds.append(X[start_index:end_index])
        start_index = end_index
        end_index += k
    
    return k_folds


# Get freq. of pair of consecutive bytes in list of tokens
# tokens is a list of tokens
# token is a list of bytes
def get_freq_pairs(tokens):
	pairs = defaultdict(int)

	for t in tokens:
		for i in range(len(t)-1):
			pairs[(t[i], t[i+1])] += 1
	return pairs

# Merge most frequent pair in all tokens
def merge_pair(tokens, pair):
	
	for i in range(len(tokens)):
		t = []
		added = 0
		for j in range(len(tokens[i])-1):
			if added==1:
				added = 0
				continue	
			if tokens[i][j]==pair[0] and tokens[i][j+1]==pair[1]:
				t.append("".join(pair))
				added = 1
			else:
				t.append(tokens[i][j])
		if added==0:
			t.append(tokens[i][-1])
		tokens[i] = t
	return tokens

# Find k most important bytes
def get_imp_bytes(tokens, k):
	bytes_ = defaultdict(int)	
	for t in tokens:
		for i in t:
			if len(i) > 1:
				bytes_[i] += 1
	sort_bytes = sorted(bytes_.items(), key=lambda x: x[1], reverse=True)
	k_imp_bytes = []
	count = 0
	# print(sort_bytes)
	for i in sort_bytes:
		if count >= k:
			break 
		count += 1
		k_imp_bytes.append(i[0])
	return k_imp_bytes

def generate_sub_words(words):
	tokens = []
	for w in words:
		t = []
		t[:0] = w
		tokens.append(t)
	num_merges = 20 # hyperparameter
	thres_freq = 10
	for i in range(num_merges):
		pairs = get_freq_pairs(tokens)
		if not pairs:
			break
		best_pair = max(pairs, key=pairs.get)
		# print(best_pair)
		if pairs[best_pair] > thres_freq:
			tokens = merge_pair(tokens, best_pair)
		else:
			break
	k = 30
	print(tokens)
	sub_words = get_imp_bytes(tokens, k)
	return sub_words

words = nltk.corpus.brown.words()
unique_words = np.unique(words)
sub_words = generate_sub_words(unique_words)
print(sub_words)