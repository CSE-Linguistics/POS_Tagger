import numpy as np
from collections import defaultdict
import re, nltk

def k_fold_data(k : int, X: np.array):
	start_index = 0
	end_index = int(np.floor(len(X)/ k))
	incrementor = int(np.floor(len(X)/ k))
	k_folds = []

	for i in range(k-1):
		k_folds.append(X[start_index:end_index])
		start_index = end_index
		end_index += incrementor
	k_folds.append(X[start_index:])
	
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
			if i.lower().endswith(j):
				val = 1
				break
		li.append(val)
	return li
def check_if_prefix(words, conditions):
	li = []
	for i in words:
		val = 0
		for j in conditions:
			if i.lower().startswith(j):
				val = 1
				break
		li.append(val)
	return li

def check_if_substr(words, conditions):
	li = []
	for i in words:
		val = 0
		for j in conditions:
			if i.lower().find(j) != -1:
				val = 1
				break
		li.append(val)
	return li

def check_if_capital_letter(words):
	li =[]
	for word in words:
		val = 0
		if any(x.isupper() for x in word): 
			val = 1
		li.append(val)
	return li
def unigram_generator(tagged_sentences):
	unigram = {}
	univ_tag_set = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
	indices = {k:v for v,k in enumerate(univ_tag_set)}
	for sentence in tagged_sentences:
		for i in range(1,len(sentence)):
			key = sentence[i][0]
			if key not in unigram:
				unigram[key] = np.zeros(12)
			unigram[key][indices[sentence[i][1]]] += 1

	for key in unigram:
		arr = unigram[key]
		arr/= np.sum(arr)
		unigram[key] = arr
	return unigram

def bigram_generator(tagged_sentences):
	bigram = {}
	univ_tag_set = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
	indices = {k:v for v,k in enumerate(univ_tag_set)}
	for sentence in tagged_sentences:
		for i in range(1,len(sentence)):
			conc_tuple = (sentence[i-1][1], sentence[i][0])
			if conc_tuple not in bigram:
				bigram[conc_tuple] = np.zeros(12)
			bigram[conc_tuple][indices[sentence[i][1]]] += 1

	for key in bigram:
		arr = bigram[key]
		arr/= np.sum(arr)
		bigram[key] = arr
	return bigram

def trigram_generator(tagged_sentences):
	trigram = {}
	univ_tag_set = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
	indices = {k:v for v,k in enumerate(univ_tag_set)}
	for sentence in tagged_sentences:
		for i in range(2,len(sentence)):
			conc_tuple = (sentence[i-2][1], sentence[i-1][1], sentence[i][0])
			if conc_tuple not in trigram:
				trigram[conc_tuple] = np.zeros(12)
			trigram[conc_tuple][indices[sentence[i][1]]] += 1

	for key in trigram:
		arr = trigram[key]
		arr/= np.sum(arr)
		trigram[key] = arr
	return trigram

def word_feature(word, trigram, bigram, unigram, previous_word_tag = None, prev_2_word_tag = None):
	vec = np.ones(12)
	vec/=12
	if previous_word_tag is None:
		if word not in unigram:
			return vec
		else:
			return unigram[word]
	if prev_2_word_tag is None:
		if (previous_word_tag, word) not in bigram:
			if word not in unigram:
				return vec
			else: return unigram[word]
		else: 
			return bigram[(previous_word_tag, word)]

	if (prev_2_word_tag, previous_word_tag, word) not in trigram:
		if (previous_word_tag, word) not in bigram:
			if word not in unigram:
				return vec
			else: return unigram[word]
		else: 
			return bigram[(previous_word_tag, word)]
	return trigram[(prev_2_word_tag, previous_word_tag, word)]


def trainFeatures(sentences):
	trigram = trigram_generator(sentences)
	bigram = bigram_generator(sentences)
	unigram = unigram_generator(sentences)
	features = []
	for sentence in sentences:
		for i in range(len(sentence)):
			if i == 0:
				features.append(word_feature(sentence[i][0], trigram, bigram, unigram))
			if i == 1:
				features.append(word_feature(sentence[i][0], trigram, bigram, unigram, previous_word_tag= sentence[i-1][1]))
			if i > 1:
				features.append(word_feature(sentence[i][0], trigram, bigram, unigram, previous_word_tag= sentence[i-1][1], prev_2_word_tag=sentence[i-2][1]))
	features = np.asarray(features)
	return features, trigram, bigram, unigram


				



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
	has_capital_feature = check_if_capital_letter(words)
	NUM_FEATURES = 10
	NUM_WORDS = 5
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
		feat.append(has_capital_feature[index])
		feat = np.asarray(feat)
		return feat
	features = np.zeros((len(words),NUM_FEATURES*NUM_WORDS))
	features[:,0] = 1
	for i in range(len(words)):
		indices = [i-2,i-1,i,i+1,i+2]
		for j in range(len(indices)):
			if(indices[j]>= 0 and indices[j] < len(words)):
				extracted_features = extract_features(i-2)
				if(extracted_features.shape[0] <NUM_FEATURES): print("HUGE ERROR!")
				features[i,j*NUM_FEATURES:(j+1)*NUM_FEATURES] = extracted_features
	return features
		
	#Once Done with Feature Set 1, move on to Feature Set 2

if __name__ == "__main__":
	words_tags = nltk.corpus.brown.tagged_sents(tagset = "universal")
	
	tags = []
	for sentence in words_tags:
		for wt in sentence:
			if(wt[1] not in tags):
				tags.append(wt[1])
	print(tags)
			
