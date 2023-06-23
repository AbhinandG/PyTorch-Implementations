import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#Defining our Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_labels):
        super(NeuralNetwork,self).__init__()
        self.num_inputs=input_size
        self.hidden_size=hidden_size
        self.output_labels=output_labels
        self.Linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.Linear2=nn.Linear(hidden_size,hidden_size)
        self.relu=nn.ReLU()
        self.Linear3=nn.Linear(hidden_size,output_labels)
        
    def forward(self,x):
        out=self.Linear1(x)
        out=self.relu(out)
        out=self.Linear2(out)
        out=self.relu(out)
        out=self.Linear3(out)
        return out
    
input_size=28*28
hidden_size=100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.001


#Loading the Data with Training and Testing Sets
train_dataset=torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(), download=True)

#Creating the DataLoader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                         shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                       shuffle=False)


#Defining the Model, Loss Function and Optimizer
model=NeuralNetwork(input_size,hidden_size,num_classes)
lossFunc=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 100,1,28,28 needs to be reshaped to 100,784
        images=images.reshape(-1,28*28)
        
        #forward pass
        outputs=model.forward(images)
        loss=lossFunc(outputs, labels)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("BATCH NUMBER: "+str(i)+" | LOSS: "+str(loss.item()))


#Testing the Model
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28)
        outputs=model.forward(images)
        _,predictions=torch.max(outputs, 1) #Returns value (_) and the index (predictions)
        n_samples+=labels.shape[0]
        n_correct+=(predictions==labels).sum().item()
    acc= (100.0) * (n_correct/n_samples)
    print('Testing Accuracy = '+str(acc))