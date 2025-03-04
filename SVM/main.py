import nltk
import model
import utils
import numpy as np
from matplotlib import pyplot as plt
import os
if __name__ == "__main__":
    words_tags = nltk.corpus.brown.tagged_sents(tagset = "universal")

    tagset = ['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
    indices = {k:v for v,k in enumerate(tagset)}

    data_list = utils.k_fold_data(5, words_tags)

    for i in range(5):
        #Step 1: Create the train-test array

        XY_train = data_list[:i] + data_list[i+1:]
        XY_train_final = XY_train[0]
        for XY in XY_train:
            XY_train_final += XY
        XY_train = XY_train_final
        XY_test  = data_list[i]
        Y_train = np.asarray([indices[wordtag[1]] for sentence in XY_train for wordtag in sentence ])


        X_train, trigram, bigram, unigram = utils.trainFeatures(XY_train)



        # #Step 2, make the SVM model
        learning_rate = 1e-1
        reg = 1e-2
        svm_model = model.MultiClassSVM(len(tagset), 22, reg = reg)
        path = f"reg_{reg}"
        os.makedirs(path,exist_ok =True)

        train_loss, W = svm_model.fit(X_train, Y_train,lr = learning_rate, epochs= 500,batch_size= X_train.shape[0])
        np.save(os.path.join(path,f"W_{i}"),W)

        #Step 3. For testing:
        confusion_matrix = np.zeros((12,12))

        acc = 0
        tot_count = 0
        for sentence in XY_test:
            prev_tag = None
            prev_2_tag = None
            count = 0
            correct_count = 0
            for wordtag in sentence:
                word = wordtag[0]
                x_test = utils.word_feature(word, trigram, bigram, unigram,prev_tag, prev_2_tag)
                predicted_class = svm_model.predict_single(x_test)
                predicted_class = predicted_class[0]                
                if(tagset[predicted_class] == wordtag[1]):
                    correct_count+=1
                confusion_matrix[indices[wordtag[1]], predicted_class]+=1
                count+=1
                prev_2_tag = prev_tag
                prev_tag = tagset[predicted_class]

                
            tot_count += count
            acc+= correct_count
        
        np.save(os.path.join(path,f"conf_mat_{i}"), confusion_matrix)
        print("Confusion Matrix:")
        print(np.array2string(confusion_matrix))
        print(f"Accuracy for k :{i} = {acc/tot_count}")
        plt.figure()
        plt.title(f"Loss for Trial No: {i+1} ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(train_loss)
        plt.savefig(os.path.join(path, f"train_loss_{i}.png"))
        diag_elem = np.diag(confusion_matrix)
        sum_pos = np.sum(confusion_matrix,axis= 1)
        for i in range(12):
            if sum_pos[i] == 0 and diag_elem[i] == 0:
                acc = 1
            else:
                acc = diag_elem[i]/sum_pos[i]
            print(f"Accuracy for {tagset[i]} = {acc}")
            
