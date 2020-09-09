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

def check_if_suffix(words, condition):
	li = []
	for i in words:
		val = 0
		for j in condition:
			if i.endswith(j):
				val = 1
				break
		li.append(val)
	return li
def check_if_prefix(words, conditions):
	li = []
	for i in words:
		val = 0
		for j in conditions:
			if i.endswith(j):
				val = 1
				break
		li.append(val)
	return li

def check_if_substr(words, conditions):
	li = []
	for i in words:
		val = 0
		for j in conditions:
			if i.find(j) != -1:
				val = 1
				break
		li.append(val)
	return li

def genFeatures(words):
	SUFFIX_NOUN = ["eer", "er", "ion", "ity", "ment", "ness", "or", "sion", "ship", "th"]
	SUFFIX_ADJECTIVE = ["able", "ible", "al", "ant", "ary", "ful", "ic", "ious", "ous", "ive", "less", "y"]
	SUFFIX_VERB = ["ed", "en", "er", "ing", "ize", "ise"]
	SUFFIX_ADVERB = ["ly", "ward", "wise"]
	HYPHEN = ["-"]
	PREFIX_NOUN = ["non", "pre"]
	PREFIX_VERB = ["dis", "mis", "ob", "op", "pre", "un", "re"]
	PREFIX_ADJECTIVE = ["anti", "en", "il", "im", "in", "ir", "non", "pre", "un"]

	#Feature Set 1 -> Word-2, Word-1, Word, Word+1, Word+2 's features
	suf_noun_feature = check_if_suffix(words, SUFFIX_NOUN)
	suf_verb_feature = check_if_suffix(words, SUFFIX_VERB)
	suf_adj_feature = check_if_suffix(words, SUFFIX_ADJECTIVE)
	suf_adverb_feature = check_if_suffix(words, SUFFIX_ADVERB)

	hyph = check_if_substr(words, HYPHEN)
	
	pref_noun_feature = check_if_prefix(words, PREFIX_NOUN)
	pref_verb_feature = check_if_prefix(words, PREFIX_VERB)
	pref_adj_feature = check_if_prefix(words, PREFIX_ADJECTIVE)
	is_alpha_feature = [word.isalpha() for word in words]
	is_alpha_feature = list(map(int,is_alpha_feature))
	## For word 1
	def extract_features(index):
		feat = []
		feat.append(suf_noun_feature[index])
		feat.append(suf_adj_feature[index])
		feat.append(suf_verb_feature[index])
		feat.append(suf_adverb_feature[index])
		feat.append(hyph[index])
		feat.append(pref_noun_feature[index])
		feat.append(pref_adj_feature[index])
		feat.append(pref_verb_feature[index])
		feat.append(is_alpha_feature[index])
		feat = np.asarray(feat)
		return feat
	features = np.zeros((len(words),45))
		
	for i in range(len(words)):
		indices = [i-2,i-1,i,i+1,i+2]
		for j in range(len(indices)):
			if(indices[j]>= 0 and indices[j] < len(words)):
				extracted_features = extract_features(i-2)
				if(extracted_features.shape[0] <9): print("HIGE ERROR!")
				features[i,j*9:(j+1)*9] = extracted_features
	
	return features
		
	#Once Done with Feature Set 1, move on to Feature Set 2

if __name__ == "__main__":

	words = nltk.corpus.brown.words()
	features = genFeatures(words)
	# unique_words = np.unique(words)
	# sub_words = generate_sub_words(unique_words)
	# print(sub_words)
