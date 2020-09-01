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
        