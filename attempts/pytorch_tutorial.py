import torch
a=torch.tensor([2.,3.], requires_grad=True)
b=torch.tensor([6.,4.], requires_grad=True)
f=3*a**3 - b**2

print(f)

f.backward(gradient=torch.tensor([1,1]))
print(a.grad)

import torch.nn as nn # layers and stuff
import torch.nn.functional as F # activation functions
import torch.optim as optim # optimizer i guess to minimize loss

from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transofmr(X_test)

num_features = 1
neurons = 3

# inherit from nn.Module, create a class
class BCNet(nn.Module):
    def __init__(self):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(num_features, neurons) # linear layer
        self.fc2 = nn.Linear(num_features, neurons-1) # linear layer
        self.fc3 = nn.Linear(neurons-1, 1)
    
    # forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x)) # feed input through first layer and the activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

model = BCNET() # model is our class
criterion = nn.BCELoss()
optimizer = optim.Adam(mode.parameters(), lr=0.001) # adam optimizer and learning rate 0.001
# pass in parameters to optimize

epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss was {running_loss / len(train_loader)}')





