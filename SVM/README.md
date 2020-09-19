# Support Vector Machines

## Steps to Run Brown Corpus Train-Test 
python3 main.py

## Implemetation Details
We have used Trigram, Bigram and Unigram probabilites, along with information regarding whether the prefix or suffix would classify the word as a particular part of speech. A lower N-gram is being used when a higher N-gram is not available. The frequencies of N-grams are then converted into ratios of chances of occurence to iitate probability scores.  
Some further heurstics like converting all digits to 1, using details regarding hyphenation, whether the word being provided just consistas of alphabets and whether there are any capitalizations. These have been addded to juice up the accuracy of Nouns and numerals. Adding these also improved accuracies of verbs, which is probably due to better classification in generals.


## References
CS 231n of Stanford: https://cs231n.github.io/linear-classify/  
NLTK N-Gram tagger: http://www.nltk.org/book/ch05.html  
Prefix and Suffix: https://web2.uvcs.uvic.ca/courses/elc/sample/beginner/gs/gs_55_1.htm  
