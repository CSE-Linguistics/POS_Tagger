# Pos tagging using Bi-LSTM

### Steps for pre-processing:
* Add sentence beginner to annotated data
* Add two words to the vocabulary - "OOV" (out of vocab.), "PAD" (word to pad the sentence)
* Add a dummy tag - "PAD" for padding
* Every unique word should have a unique label
* Every unique POS tag should have a unique label
* Every tag should be coded into one-hot vectors

### Design of the model:
* The layers of the model are as follows-
<center>
	|:-----------------:|
	|  Embedding layer  |
	|Bidirectional layer|
	|    Dense layer    |
	|Softmax activation |
</center>

### Some important points:
* One should take care to ignore the accuracy for predicting the dummy tag - "PAD" while training the model
* We should also not consider the tag - "PAD" while evaluating the model's performance on test data
* This model is end-to-end and requires no pre-trained word vectors or feature engineering