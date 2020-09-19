# POS Tagging Using Hidden Markov Model

We used NLTK's brown corpus to train and test our model. The tagset used was Universal Tagset consisting of 12 different tags. 

### To Run

python3 hmm.py

### Pre-Processing

* Add sentence beginner to annotated data
* Replaced consecutive sequence of numbers by 0, for e.g, 123AB12 -> 0AB0

> For unseen words, emission probability is taken as 1 for all tags

### Results

* The accuracy obtained on test data is close to 95.2%
* We evaluated per-POS accuracy and found that words with tags having very occurrences(e.g. X) or unseen words were predictably incorrectly more often as compared to others