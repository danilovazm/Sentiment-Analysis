import torch
import torch.nn as nn
import torch.optim as optim
from Network import LSTM, init_hidden

def Train(train_loader, rnn, epochs, lr, batch_size, device):
    rnn.to(device)
    Loss = nn.BCELoss()
    optimizer = optim.Adam(rnn.parameters(), lr)
    hidden = init_hidden(100, device)
    train_losses = []
    accu = 0
    print(type(epochs))
    for i in range(epochs):
        rnn.train(True)
        totalLoss = 0
        j = 0
        for iter, (x ,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)       
            prediction, hidden = rnn(x, hidden)
            prediction = prediction.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            y = y.detach()
            hidden = (hidden[0].detach(), hidden[1].detach())
            if torch.round(prediction) == y:
                j +=1 
            loss = Loss(prediction, y)
            rnn.zero_grad()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            totalLoss += loss.item()
        accu = j/len(train_loader)
        print("Epoca: " + str(i+1) + " Loss: " + str(totalLoss/len(train_loader)) + "Acuracia: " + str(accu))
        train_losses.append(totalLoss/len(train_loader))
    return train_losses, rnn, hidden