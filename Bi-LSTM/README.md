# POS tagging using Bi-LSTM

We used NLTK's brown corpus to train and validate our model. The tagset used was Universal tagset consisting of 12 different tags. To implement the model we used Tensorflow 2.0.

### Steps for pre-processing:
* Add sentence beginner to annotated data
* Add two words to the vocabulary - "OOV" (out of vocab.), "PAD" (word to pad the sentence)
* Add a dummy tag - "PAD" for padding
* Every unique word should have a unique label
* Every unique POS tag should have a unique label
* Tags should be coded into one-hot vectors

### Design of the model:
* The layers of the model are as follows-

	|     Layers        |
	|:-----------------:|
	|  Embedding layer  |
	|Bidirectional layer|
	|    Dense layer    |
	|Softmax activation |

### Some important points:
* One should take care to ignore the accuracy for predicting the dummy tag - "PAD" while training the model
* We should also not consider the tag - "PAD" while evaluating the model's performance on test data
* This model is end-to-end and requires no pre-trained word vectors or feature engineering

### Results
* The accuracy obtained on test data only after 2 epochs of training was 95.8%
* We evaluated the per-POS accuracy and found that the tags having less occurrences are predicted poorly as compared to others
* We also found that many wrong predictions were due to unseen words in the test data