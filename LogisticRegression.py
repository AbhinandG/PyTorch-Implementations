import torch
import numpy as np
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self,X,Y):
        super(LogisticRegression,self).__init__()
        self.X=X
        self.Y=Y
        self.n_samples=X.shape[0]
        self.n_features=X.shape[1]
        self.model=nn.Linear(self.n_features,1)
        self.lossFunc=nn.BCELoss() #Binary Crossentropy
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=0.1)
        
    def forward(self,x):
        return torch.sigmoid(self.model(x))
    
    def train(self, epochs):
        X_train,X_test,y_train,y_test=train_test_split(self.X,self.Y,test_size=0.25,random_state=1)
        sc=StandardScaler()
        X_train=sc.fit_transform(X_train)
        X_test=sc.transform(X_test)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        y_train=y_train.view(y_train.shape[0],1)
        y_test=y_test.view(y_test.shape[0],1)
        
        
        print("========== STARTING TO TRAIN ============")
        
        for epoch in range(epochs):
            #print("......going for epoch number - "+str(epoch+1)+".......")
            y_predicted=self.forward(X_train)
            #print("The y_predicted here is - "+str(y_predicted))
            L=self.lossFunc(y_predicted,y_train)
            L.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print("====EPOCH #"+str(epoch+1)+": Loss="+str(L.item()))
            
            
        print("========== TRAINING COMPLETED ============")
        
        
        with torch.no_grad():
            y_predicted=torch.sigmoid(self.model(X_test))
            y_predicted_cls=y_predicted.round()
            acc = ((y_predicted_cls == y_test).sum().item() / float(y_test.shape[0]))
            print("ACCURACY OF THE MODEL - "+str(acc)+"%")