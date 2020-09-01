import numpy as np

class SupportVectorMachine:
    def __init__(self, num_features:int,reg:float):
        self.W = np.random.normal(0,1,(num_features,1))
        self.b = np.random.normal(0,1,1)        
        self.reg = reg
    def predict(self, X:np.array):
        raw_score = X@self.W + self.b
        raw_score[raw_score < 0] = -1
        raw_score[raw_score >= 0] = 1
        return raw_score

    def evaluate(self, X: np.array, Y: np.array):
        raw_score = X@self.W + self.b
        modulo_case = Y * raw_score
        each_loss = np.zeros(Y.shape)
        each_loss[modulo_case <1 ] = 1 - modulo_case

        raw_score[raw_score < 0] = -1
        raw_score[raw_score >= 0] = 1
        loss = self.reg * np.sum(self.W *self.W) + np.sum(each_loss)

        accuracy = np.sum(raw_score == Y)/Y.shape[0]


        return raw_score, loss
    
    
    def train(self, X_train: np.array, Y_train: np.array, X_validate: np.array, Y_validate : np.array, epochs: int = 2000,lr:int = 0.01):
        train_losses, validate_losses = [], []
        for i in range(epochs):
            raw_score = X_train @ self.W + self.b
            modulo_case = Y_train * raw_score
            each_loss = np.zeros(Y_train.shape)
            each_loss[modulo_case <1 ] = 1 - modulo_case

            raw_score[raw_score < 0] = -1
            raw_score[raw_score >= 0] = 1
            loss = self.reg * np.sum(self.W *self.W) + np.sum(each_loss)
            train_losses.append(loss)
            validate_scores, validate_loss = evaluate(X_validate, Y_validate)
            validate_losses.append(validate_loss)
            W -= (2*self.reg)*W
            y_train_cpy = np.copy(Y_train)
            y_train_cpy[each_loss  == 0] = 0
            X_train_new = np.copy(X_train)
            X_train_new = X_train_new*y_train_cpy
            W += lr*(np.sum(X_train_new,axis = 0) - (2*self.reg)*W)

            if i %100 == 0:
                print(f'Losses at step {i}: Train loss: {train_losses[-1]}, Validate Loss : {validate_losses[-1]}')
            

        return train_losses, validate_losses
        
        
        pass
        
class MultiClassSVM:
    def __init__(self, num_classes:int, num_features:int, reg:float, delta: float = 1.0):
        self.W = np.random.normal(0 ,1, (num_features, num_classes))
        self.B = np.random.normal(0,1,(1,num_features))
        self.reg = reg
        self.d = delta

    
    def predict(self, X:np.array):
        raw_score = X@self.W + self.B
        max_col = np.argmax(raw_score, axis = 1)
        final_score = np.zeros((raw_score.shape[0],W.shape[1]))


        return max_col

    def evaluate (self, X: np.array, Y: np.array):
        num_data = X.shape[0]
        num_classes = self.W.shape[1]
        raw_score = X@self.W + self.B
        correct_answer_score = np.copy(raw_score[np.arange(num_data), Y])
        max_col = np.argmax(raw_score, axis = 1)
        accuracy = np.sum((max_col == Y))/num_data

        loss = raw_score - correct_answer_score[:, np.newaxis] + self.d
        loss[loss< 0 ] = 0 
        loss[np.arange(num_data), Y] = 0
        loss_array = np.copy(loss)
        loss = np.sum(loss)/num_data + self.reg*np.linalg.norm(self.W)**2

        return max_col, accuracy, loss, loss_array
    
    def fit(self, X: np.array, Y: np.array, X_test:np.array, Y_test:np.array, lr :float = 0.01, epochs: int = 1000):
        num_data = X.shape[0]
        num_classes = self.W.shape[1]
        raw_score = X @ self.W + self.B
        train_losses = []
        test_losses = []


        #-----------Gradient Descent-----------------------
        for i in range(epochs):
            Y_pred, acc, train_loss, loss_arr = evaluate(X, Y)
            Y_pred_test, acc_test, test_loss, _ = evaluate(X_test, Y_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            #--------------For W--------------------------
            num_xi = np.sum(loss_arr > 0, axis = 1)
            loss_arr[loss_arr > 0] =1 
            val_xi = num_xi[:,np.newaxis] * X
            zeros_arr = np.zeros((num_data,num_classes))
            zeros_arr[np.arange(num_data),Y] = 1
            dW = - val_xi.T @ zeros_arr
            dW += X.T @ loss_arr
            dW/= num_data
            dW += 2*self.reg*self.W
            #------------For B---------------------------------
            dB = np.sum(loss_arr,axis = 1)
            num_bins = np.bincount(num_xi)
            dB [:num_bins.shape[0]] = num_bins
            dB/= num_data
            dB = dB[:,np.newaxis]
            #-------------LR Step------------------------------
            self.W -= lr*dW
            self.B -= lr*dB
            #------------Print Loss-----------------------------
        
        return train_losses, test_losses, self.W, self.B