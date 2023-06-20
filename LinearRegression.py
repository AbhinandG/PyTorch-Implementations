import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, X, Y, epochs):
        super(LinearRegression, self).__init__()
        self.X=X
        self.Y=Y
        self.n_samples=X.shape[0]
        self.n_features=X.shape[1]
        self.input_size=n_features
        self.output_size=n_features
        self.model=nn.Linear(input_size, output_size)
        self.lossFunc=nn.MSELoss()
        self.learning_rate=0.1
        self.epochs=epochs
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
    def forward(self, X):
        return self.model(X)
    
    def train(self):
        for epoch in range(self.epochs):
            y_pred=self.model.forward(self.X)
            l=lossFunc(y_pred,self.Y)
            l.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print("Epoch #"+str(epoch+1)+" Testing for F(25): Value="+str(self.model.forward(torch.tensor([[25]],dtype=torch.float32))))

        