import nltk
import model
import utils
import numpy as np
from matplotlib import pyplot as plt
if __name__ == "__main__":
    words_tags = nltk.corpus.brown.tagged_words(tagset='universal')
    words = [word_tag[0] for word_tag in words_tags]
    tags_for_words = [word_tag[1] for word_tag in words_tags]
    tagset = list(set(tags_for_words))
    numbering = {word_tag[1]:word_tag[0] for word_tag in enumerate(tagset)}
    numbered_tags = [numbering[tag] for tag in tags_for_words]
    numbered_tags = np.asarray(numbered_tags)
    ones = np.ones(len(words))
    ones = ones[:, np.newaxis]
    features = utils.genFeatures(words)
    features = np.hstack((ones, features))
    data_list = utils.k_fold_data(5, features)
    result_list = utils.k_fold_data(5, numbered_tags)
    svm_model = model.MultiClassSVM(len(tagset), data_list[0].shape[1])
    train_losses = []
    test_losses = []
    for i in range(5):
        #Step 1: Create the train-test array
        X_train = data_list[:i] + data_list[i+1:]
        X_train = np.concatenate(X_train)
        X_test  = data_list[i]
        Y_train = result_list[:i] + result_list[i+1:]
        Y_test = result_list[i]
        Y_train = np.concatenate(Y_train)

        #Step 2, make the SVM model
        train_loss, test_loss, W = svm_model.fit(X_train, Y_train, X_test, Y_test,lr = 0.01, epochs= 10000)
        train_losses += train_loss
        test_losses += test_loss
    plt.plot(train_losses)
    plt.plot(test_losses)

    plt.show()
    
    np.save("W.npz",W)

