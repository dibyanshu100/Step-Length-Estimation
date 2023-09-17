# Python file to store all models
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch



class ModelMLP:
    def __init__(self,in_dim,hidden_dim,class_dim,lr) -> None:
        self.model = nn.Sequential(
    nn.Linear(in_dim, hidden_dim, bias=True),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim, bias=True),
    nn.ReLU(),
    nn.Linear(hidden_dim, class_dim, bias=True),
    nn.Sigmoid() # using sigmoid since the data is in zeros and ones
    )

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss_func = nn.CrossEntropyLoss()

class CustomGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(CustomGRU,self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return self.sigmoid(out)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden


class ModelGRU:
    def __init__(self, input_size, output_size, hidden_dim, n_layers,lr)->None:
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.model = CustomGRU(input_size=input_size,output_size=output_size,hidden_dim=hidden_dim,n_layers=n_layers) 

        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.loss_func = nn.MSELoss()

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,dropout_rate):
        super(LSTM,self).__init__()
    
        self.hidden_size = hidden_size

        # Define the LSTM Layer 1
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Define the LSTM Layer 2
        self.lstm2 = nn.LSTMCell(input_size,hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Define the output layer
        self.fc = nn.Linear(hidden_size, output_size)

        # Apply sigmoid activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):

        # Forward pass through the LSTM Layer

        out1,_ = self.lstm1(input)
        out1  = self.dropout1(out1)

        out2,_ = self.lstm2(out1)
        out2 = self.dropout2(out2)

        # Forward pass through the output layer
        #print(out2[0][:,-1])
        # print(out2[:,-1,:])
        #print(out2)
        out = self.fc(out2)

        return self.sigmoid(out)


class ModelLSTM:
    def __init__(self, input_size, hidden_size, output_size,dropout_rate,lr) -> None:
        
        self.model = LSTM(input_size=input_size,hidden_size=hidden_size,output_size=output_size,dropout_rate=dropout_rate)

        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)




def getModels(modelName):
    "Return model"
    m1 = ModelMLP(6,64,2,0.01)
    m2 = ModelGRU(input_size=6,output_size=2,hidden_dim=64,n_layers=2,lr=0.001)
    m3 = ModelLSTM(input_size=6,hidden_size=6,output_size=2,dropout_rate=0.2,lr=0.001)

    if modelName == 'm1': return m1
    elif modelName == 'm2': return m2
    elif modelName == 'm3': return m3





def plot(metric1,metric2,m1_label,m2_label,x_label,y_label,title):
    plt.plot(metric1,label = m1_label)
    plt.plot(metric2, label = m2_label)

    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)

    plt.legend()
    plt.show()


def trainModel(model,train_loader,epochs):
    train_loss_per_epoch = []
    #print(f"Dataset size : {len(train_loader)}")
    for epoch in range(epochs):
        train_loss = 0.0

        for inputs,labels in train_loader:

            #Zero the gradients
            model.optimizer.zero_grad()

            # Forward pass
            outputs = model.model(inputs)

            # Compute the loss
            loss = model.loss_func(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the model parameters
            model.optimizer.step()

            train_loss+=  loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss / len(train_loader):.4f}")

        train_loss_per_epoch.append(train_loss/len(train_loader))
    
    return (model, train_loss_per_epoch)

        

