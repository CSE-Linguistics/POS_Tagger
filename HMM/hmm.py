#!/usr/bin/env python
# coding: utf-8

# In[35]:


import re
import numpy as np
import sklearn
from nltk.corpus import brown

def mylog(x):
    if x==0:
        return -1e5
    else:
        return np.log10(x)

class HMM:
    
    
    #Preprocess dataset: train and test
    def __init__(self, train_data, test_data):
        
        '''
            train_corpus: processed train_data
            test_corpus : processed test_data
        '''
        
        self.train_corpus = []
        for i in range(len(train_data)):
            sent = []
            sent.append(("^", "^"))
            sent = sent + [(re.sub('\d{1,}', '0', w), t) for (w, t) in train_data[i]]
            self.train_corpus.append(sent)
        
        self.test_corpus = []
        for i in range(len(test_data)):
            sent = []
            sent.append(("^", "^"))
            sent = sent + test_data[i]
            self.test_corpus.append(sent)
        
    #Train HMM
    def train(self):
        '''
            all_tags: contains all tags in the corpus
            tags_freq: frequency of individual tags
            tags_to_num: tags to index
            num_to_tags: index to tags
            all_words: all words in the corpus
            transition: transition probabilities
            emission: emission probabilities
        '''
        
        #All tags in the train corpus
        self.all_tags = []
        for sent in self.train_corpus:
            for (w, t) in sent:
                if t not in self.all_tags:
                    self.all_tags.append(t)
        
        self.all_tags.sort()
        #Count frequency for each tag
        self.tags_freq = {}
        for t in self.all_tags:
            self.tags_freq[t] = 0
        for sent in self.train_corpus:
            for (w, t) in sent:
                self.tags_freq[t] += 1
        
        #length of corpus
        self.total_tag = 0
        for t in self.all_tags:
            self.total_tag += self.tags_freq[t]

        #tag to number and vice-versa
        self.tags_to_num = {}
        self.num_to_tags = {}
        for i in range(len(self.all_tags)):
            self.tags_to_num[self.all_tags[i]] = i
            self.num_to_tags[i] = self.all_tags[i]

        #All possible words in the set
        self.all_words = set()
        for sent in self.train_corpus:
            for (w, t) in sent:
                self.all_words.add(w)

        #Count frequency for each (word, tag)
        word_tag_freq = {}
        for w in self.all_words:
            for t in self.all_tags:
                word_tag_freq[(w, t)] = 0
        for sent in self.train_corpus:
            for (w, t) in sent:
                word_tag_freq[(w, t)] += 1

        #evaluate transition  counts
        self.transition = {}
        for t1 in self.all_tags:
            for t2 in self.all_tags:
                self.transition[(t1, t2)] = 0
        for sent in self.train_corpus:
            for i in range(len(sent)-1):
                self.transition[(sent[i][1], sent[i+1][1])] += 1

        #evaluate transition and emission probabilities
        self.emission = {}
        for (t1, t2) in self.transition.keys():
            self.transition[(t1, t2)] /= self.tags_freq[t1]
        for (w, t) in word_tag_freq.keys():
            self.emission[(w, t)] = word_tag_freq[(w, t)]/self.tags_freq[t]
    
    
    #Implements Viterbi Algorithm
    def viterbi(self, sent):
        '''
            sent: input sentence
            returns predicted tags
        '''
        # Replace all numbers by 0
        sentence = [re.sub('\d{1,}', '0', w) for w in sent]
            
        len_sent = len(sentence)
        len_tagset = len(self.all_tags)

        #SEQSCORE and BACKPTR arrays
        SEQSCORE = [[mylog(0) for i in range(len_sent)] for j in range(len_tagset)]
        BACKPTR =  [[0 for i in range(len_sent)] for j in range(len_tagset)]

        null_tag = self.tags_to_num["^"]
        #initialise the null tag
        SEQSCORE[null_tag][0] = 0

        for i in range(1, len_sent):#Corresponds to a given word sentence[i]
            for cidx, ctag in enumerate(self.all_tags):#Ending at current tag

                optimal_prob = -1e9 #Includes transitional probabilites
                optimal_tag = 0

                for pidx, ptag in enumerate(self.all_tags):#Previous tag
                    prob_k_j_i = SEQSCORE[pidx][i-1] + mylog(self.transition[(ptag, ctag)])
                    if prob_k_j_i > optimal_prob:
                        optimal_prob = prob_k_j_i
                        optimal_tag = pidx

                if sentence[i] in self.all_words:
                    SEQSCORE[cidx][i] = optimal_prob + mylog(self.emission[(sentence[i], ctag)])
                else:
                    SEQSCORE[cidx][i] = optimal_prob
                BACKPTR[cidx][i] = optimal_tag

        #Sequence identification step
        CT = 0
        optimal_prob = -1e9
        for i in range(len_tagset):
            if SEQSCORE[i][len_sent-1]>optimal_prob:
                optimal_prob = SEQSCORE[i][len_sent-1]
                CT = i
        
        pred_tags = [CT for i in range(len_sent)]
        for i in reversed(range(len_sent-1)):
            pred_tags[i] = BACKPTR[pred_tags[i+1]][i+1]

        pred_tags = [self.num_to_tags[idx] for idx in pred_tags]
        return pred_tags
    
    def evaluation(self):
        '''
            Returns evaluation metrics on the train_corpus
        '''
        len_tagset = len(self.all_tags)
            
        confusion_matrix = np.zeros((len_tagset, len_tagset), dtype = np.int32)
        
        for sent in self.test_corpus:
            #Untagged Sentence
            untagged_sent = []
            for (w, t) in sent:
                untagged_sent.append(w)
    
            predicted_tags = self.viterbi(untagged_sent)
            
            for i in range(len(sent)):
                confusion_matrix[self.tags_to_num[predicted_tags[i]]][self.tags_to_num[sent[i][1]]] += 1
        
        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)
        return accuracy, confusion_matrix

    
corpora = brown.tagged_sents(tagset='universal')
num_folds = 5
num_tags = 13
all_tags = {}
subset_size = int(len(corpora)/num_folds)
sum_confusion_matrix = np.zeros((num_tags, num_tags), dtype = np.int32)

for i in range(5):
    
    test_data  = corpora[i*subset_size:][:subset_size]
    train_data = corpora[:i*subset_size] + corpora[(i+1)*subset_size:]
    hmm = HMM(train_data, test_data)
    hmm.train()
    accuracy, confusion_matrix = hmm.evaluation()
    
    sum_confusion_matrix += confusion_matrix
    print("\nEvaluating Fold:{}".format(i+1))
    print("Accuracy:{}".format(accuracy))
    if i == 2:
        all_tags = hmm.all_tags
        print("\nConfusion Matrix for Fold:3")
        print(confusion_matrix)

accuracy = np.trace(sum_confusion_matrix)/np.sum(sum_confusion_matrix)
accuracy_per_pos = {}
recall_per_pos   = {}
for idx, tag in enumerate(all_tags):
    accuracy_per_pos[tag] = sum_confusion_matrix[idx][idx]/np.sum(sum_confusion_matrix[:, idx])
    recall_per_pos[tag]   = sum_confusion_matrix[idx][idx]/np.sum(sum_confusion_matrix[idx, :])
print("\nPer POS Accuracy")
print(sorted(accuracy_per_pos.items()))
print("\nPer POS Recall")
print(sorted(recall_per_pos.items()))
print("Average Accuracy:{}".format(accuracy))

